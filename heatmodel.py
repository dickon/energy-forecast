import json, pprint, pickle, os
from dateutil.parser import isoparse
from datetime import datetime, timedelta
from typing import NamedTuple
import influxdb_client, time
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd
from bokeh.layouts import column, row, layout, gridplot
from bokeh.plotting import figure, show, output_file
from bokeh.models import Slider, Switch, Div, CrosshairTool, Span, HoverTool, ColumnDataSource, Range1d, LinearAxis,  DateRangeSlider, RadioGroup, Button
from bokeh.io import curdoc
from structures import HouseData, TemperaturePoint
from bokeh.palettes import Category20, magma, plasma
from bokeh.transform import linear_cmap
from math import ceil
import scipy.integrate
import pytz

    
interval_minutes = 10

LOCATION_TRANSFORMATION = {
      'guest_suite': 'Downstairs Guest suite',
      'box': 'Box',
      'bathroom': 'Bathroom',
      'study': 'Downstairs study',
      'bathroom': 'Bathroom',
      'cave': 'Upstairs study',
      'entrance': 'Entrance hall',
      'lounge':'Lounge',
      'master_suite': 'Master suite',
      'middle_bed': 'Middle bedroom',
      'utility':'Utility room',
      'kitchen':'Kitchen'
}

LOCATION_REVERSE = {
    "Downstairs toilet": "Lounge",
    "Guest suite shower room": "Downstairs Guest suite",
    "Medal area":"Lounge",
    "Dining room":"Lounge"
}
class InfluxConfig(NamedTuple):
    token: str
    server: str
    bucket: str
    org: str

def row_split(elements, count=3):
    return list([row(elements[i*count:(i+1)*count]) for i in range(ceil(len(elements)/count)) ])

site = json.load(open(os.path.expanduser("~/site.json")))
def fetch_house_data() -> HouseData:
    influxConfig = InfluxConfig(**site["influx"])
    client = influxdb_client.InfluxDBClient(
        url=influxConfig.server, token=influxConfig.token, org=influxConfig.org
    )

    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()

    start = isoparse("2023-08-01T00:00:00Z")
    temps = []
    for col in query_api.query(f"""
from(bucket: "home-assistant")
  |> range(start:{start.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {(start+timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')})
  |> filter(fn: (r) => r["_measurement"] == "sensor.openweathermap_condition" or r["_measurement"] == "weather.56_new_road")
  |> filter(fn: (r) => r["_field"] == "temperature_valid" or r["_field"] == "temperature")
  |> filter(fn: (r) => r["domain"] == "weather")
  |> filter(fn: (r) => r["entity_id"] == "56_new_road")
    """        
    ):
        for row in col:
            t = TemperaturePoint(row['_time'], row["_value"])
            temps.append(t)
    room_temperatures = {}
    room_setpoints = {}
    for col in query_api.query(f"""
from(bucket: "56new")
  |> range(start:{start.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {(start+timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')})
  |> filter(fn: (r) => r["_measurement"] == "temperature" or r["_measurement"] == "temperature_direct")
  |> filter(fn: (r) => r["_field"] == "temperature" or r["_field"] == "setpoint")
  |> aggregateWindow(every: {interval_minutes}m, fn: mean, createEmpty: false)
  |> yield(name: "mean")
        """):
        for row in col:
            loc = LOCATION_TRANSFORMATION.get(row['location'], row['location'])
            
            if row['_field'] == 'temperature':
                room_temperatures.setdefault(loc, {})
                room_temperatures[loc][row["_time"]] = row["_value"]
            if row['_field'] == 'setpoint':
                room_setpoints.setdefault(loc, {})
                room_setpoints[loc][row['_time']] = row['_value']
    gas_readings = {}
    for col in query_api.query(f"""
        from(bucket: "56new")
        |> range(start:{start.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {(start+timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')})
        |> filter(fn: (r) => r["_measurement"] == "usage")
        |> filter(fn: (r) => r["_field"] == "usage")
        |> filter(fn: (r) => r["device_name"] == "utility meter")
        |> filter(fn: (r) => r["energy"] == "gas" )
                               """):
        for row in col:
            gas_readings[(row["_time"]+timedelta(minutes=5))] = row["_value"]
    flow_temperatures = {}
    for col in query_api.query(f"""
        from(bucket: "56new")
        |> range(start:{start.strftime('%Y-%m-%dT%H:%M:%SZ')}, stop: {(start+timedelta(days=365)).strftime('%Y-%m-%dT%H:%M:%SZ')})
        |> filter(fn: (r) => r["location"] == "house_send")
        |> aggregateWindow(every: 10m, fn: mean, createEmpty: false)
        """):
        for row in col:
            flow_temperatures[(row["_time"])] = row['_value']
    return HouseData(outside_temperatures=temps, room_setpoints=room_setpoints, room_temperatures=room_temperatures, gas_readings=gas_readings, flow_temperatures=flow_temperatures)


def load_house_data() -> HouseData:
    CACHE = f'house_data_{interval_minutes}.pickle'
    if os.path.isfile(CACHE):
        print('reading', CACHE)
        with open(CACHE, 'rb') as f:
            return pickle.load(f)
    else:
        print('not found', CACHE)
        house_data = fetch_house_data()
        with open(CACHE, 'wb') as f:
            pickle.dump(house_data, f)
        return house_data

house_data = load_house_data()
#pprint.pprint(house_data.gas_readings)

with open('heatmodel.json', 'r') as f:
    data = json.load(f)

def calculate_data(focus_room: str='', verbose: bool = False, samples=1000, real_temperatures=True) -> pd.DataFrame:
    temperatures = {}
    start_t = t = pytz.utc.localize(datetime.fromtimestamp(day_range_slider.value[0]/1000))
    end_t = pytz.utc.localize(datetime.fromtimestamp(day_range_slider.value[-1]/1000))
    radiator_scales = scale_radiators()
    
    room_records = [ (calculate_room_name_alias(room_name), room_name, room_data) for (room_name, room_data) in data['rooms'].items() ]

    cursor = 0
    recs = {}
    while t < end_t:
        while cursor+ 1< len(house_data.outside_temperatures) and house_data.outside_temperatures[cursor+1].time < t:
            cursor += 1
        rec = house_data.outside_temperatures[cursor]
        recs.setdefault('meters', []).append(house_data.gas_readings.get(t, 0.0))
        temperatures['external'] = rec.temperature
        temperatures['carport'] = max(10, rec.temperature)
        recs.setdefault('time', []).append(t)
        satisfaction = 1.0
        flow_t = calculate_flow_temperature(temperatures, t, real_temperatures)
        recs.setdefault('flow_temperature', []).append(flow_t)

        weather_compensation_ratio = 1.0
        room_powers = {}
        discrepanices = {}


        # work out the target temperature for all rooms, plus a lagged set of values to reflect system response time
        target_ts= {}
        target_ts_lagged = {}
        for room_name_alias, room_name, room_data in room_records:
            target_t, target_t_lagged = calculate_target_temperature(t, room_name_alias)
            target_ts[room_name] = target_t
            target_ts_lagged[room_name] = target_t_lagged
        recs.setdefault("setpoint", []).append(target_ts[focus_room])
        recs.setdefault("setpoint_lagged", []).append(target_ts_lagged[focus_room])

        # work out heat loss for each room
        room_flows_watts = {}
        for _, room_name, room_data in room_records:                    
            room_flows_watts[room_name] = work_out_room_flow(temperatures, room_name, room_data)
            # room flows are normally negative, so we invert the series for the chart so it is readable
            recs.setdefault(f'{room_name}_loss', []).append(-room_flows_watts[room_name])
        recs.setdefault('loss_for_room', []).append(-room_flows_watts[focus_room])
        # work out radiator output if the heat source is infinite
        recs.setdefault('available_radiator_power', []).append(calculate_available_radiator_watts(temperatures, radiator_scales, flow_t, focus_room, data['rooms'][focus_room]))
        rooms_rad_watts_unbounded = calculate_radiator_outputs(temperatures, radiator_scales, room_records, 1.0, flow_t, target_ts_lagged, room_flows_watts, t)
        recs.setdefault('gain_for_room_unbounded', []).append(rooms_rad_watts_unbounded[focus_room])
        ideal_power_out = max(500, sum(rooms_rad_watts_unbounded.values()))

        # now work out how much of unbounded heat output can actually be satisfied by the boiler
        if ideal_power_out < power_slider.value:
            satisfaction = 1.0
        else:
            satisfaction =  power_slider.value * 0.95 / ideal_power_out
        
        recs.setdefault('satisfactions', []).append(satisfaction)
        rooms_rad_watts = {k:v*satisfaction for k,v in rooms_rad_watts_unbounded.items()}
        actual_power_out = sum(rooms_rad_watts.values())
        if actual_power_out > power_slider.value:
            print('unbounded:')
            pprint.pprint(rooms_rad_watts_unbounded)
            print('bounded:')
            pprint.pprint(rooms_rad_watts)
            print('satisfaction', satisfaction, 'still using', actual_power_out, 'was', ideal_power_out,'c/w limit', power_slider.value, 'breakdown', rooms_rad_watts, 'ratio to ideal is', actual_power_out/power_slider.value)
        for _, room_name, room_data in room_records:                                                    
            room_net_flow_watts = room_flows_watts[room_name] + rooms_rad_watts[room_name]
            
            recs.setdefault(room_name+'_gain', []).append(rooms_rad_watts[room_name])
            temp_change = room_net_flow_watts * interval_minutes / 60 / (room_data['volume']*temperature_change_factor_slider.value)
            orig_temp = temperatures[room_name]
            temperatures[room_name] += temp_change
            if real_temperatures:
                room_for_temp = LOCATION_REVERSE.get(room_name, room_name)
                assert room_for_temp in house_data.room_temperatures, (room_name, house_data.room_temperatures.keys())
                realtemp = house_data.room_temperatures[room_for_temp].get(t)
                if realtemp is not None:
                    actual_temp_change = realtemp - orig_temp
                    actual_flow = actual_temp_change * (room_data['volume']*temperature_change_factor_slider.value)
                    delta = actual_flow - room_net_flow_watts
                    discrepanices[room_name] = delta
                    temperatures[room_name] = realtemp
            else:
                #print('room missing for', room_name)
                pass

            if verbose:
                print(f'   Total room net flow={room_net_flow_watts:.0f}W area {room_data['area']:.0f}qm^2 temperature +{temp_change}->{temperatures[room_name]}')
        recs.setdefault('gain_for_room', []).append(rooms_rad_watts[focus_room])
        if False or actual_power_out > power_slider.value:
            print('total rad output', actual_power_out, 'c/w', power_slider.value, 'satisfaction', satisfaction)
        
        #assert house_rad_output_watts <= power_slider.value*1.001, (house_rad_output_watts, power_slider.value, satisfaction, ideal_power_out)
        efficiency = calculate_gas_efficiency(flow_t - 20.0)
        recs.setdefault('input_power', []).append ( actual_power_out / efficiency )
        if False:
            print(f'{t} power {house_rad_output_watts_unbounded/1e3:.1f}kW (ideal:{ideal_power_out/1e3:.1f}kW) inside={temperatures['Downstairs study']:.1f}C outside={temperatures['external']:.1f}C power={house_rad_output_watts_unbounded} satisfaction={satisfaction*100:.0f}%')

        for k,v  in temperatures.items():
            recs.setdefault(k+'_temperature', []).append(v)
        recs.setdefault('temperature', []).append(temperatures[focus_room])
        recs.setdefault('power_discrepancy', []).append(discrepanices.get(focus_room,0))
        #assert house_rad_output_watts >= 0, (house_rad_output_watts, power_slider.value, satisfaction, ideal_power_out)
        recs.setdefault('output_power', []).append(actual_power_out)
        t += timedelta(minutes=interval_minutes)

    df = pd.DataFrame(recs)
    # if we have lots of data, then downsample to save network bandwidth and client load
    if len(df) > samples:
        df.set_index('time', inplace=True)
        delta = end_t - start_t
        delta_sample= delta / samples
        rdf = df.resample(delta_sample).mean()
        return rdf
    else:
        rdf = df
    if real_temperatures:
        rdf_sim = calculate_data(room, real_temperatures=False)
        rdf['simulated_temperature'] = rdf_sim['temperature']
    rdf['time_str'] = rdf['time'].apply(lambda v: v.strftime('%a %d %b %y %H:%M'))
    return rdf

def calculate_radiator_outputs(temperatures, radiator_scales, room_records, satisfaction, flow_t, target_ts, room_flows_watts, timestamp, verbose=False):
    rooms_rad_watts = {}
    for _, room_name, room_data in room_records:             
        target_t = target_ts[room_name]                           
        available_rad_watts = calculate_available_radiator_watts(temperatures, radiator_scales, flow_t, room_name, room_data)
        t = temperatures[room_name] 
        text = f'at {timestamp} room {room_name} is at {t}C target {target_t}C '
        if t < target_t-0.25:
            # room too cold; rads run flat out
            room_rad_watts= available_rad_watts*satisfaction
            if verbose:
                print(text, 'heat at', room_rad_watts)
        elif t < target_t+0.25:
            # room about right; rads run at heat output
            room_rad_watts = min(available_rad_watts*satisfaction, max(-room_flows_watts[room_name], 0))
            if verbose:
                print(text, 'maintain at', room_rad_watts, 'avail', available_rad_watts, 'satisfaction', satisfaction, 'modulated avaialble', available_rad_watts*satisfaction, 'flow', room_flows_watts[room_name])
        else:
            # room too hot; rads off
            room_rad_watts = 0
            if verbose:
                print(text, 'too hot')
        rooms_rad_watts[room_name] = room_rad_watts
        if satisfaction < 1.0:
            print(room_name, 'use', room_rad_watts, 'c/w', available_rad_watts, 'satisfaction', satisfaction)
    return rooms_rad_watts

def calculate_available_radiator_watts(temperatures, radiator_scales, flow_t, room_name, room_data):
    available_rad_watts = 0
    for rad in room_data['radiators']:
        temperatures.setdefault(room_name, 21)
        rad_delta_t = max(0, flow_t - temperatures[room_name])
        rad_watts = (rad['heat50k'] *radiator_scales[room_name] * rad_delta_t / 50)
        available_rad_watts += rad_watts
    return available_rad_watts

def work_out_room_flow(temperatures, room_name, room_data):
    room_tot_flow_watts = 0
    temperatures.setdefault(room_name, 20)
    delta_t = temperatures['external'] - temperatures[room_name]

    infiltration_watts = air_change_sliders[room_name].value * air_factor_slider.value * room_data['volume'] * delta_t
    if False:
        print(f'{room_name} infiltration={infiltration_watts} from delta T {delta_t}')
    room_tot_flow_watts += infiltration_watts
    for elem in room_data['elements']:
        try:
            id = elem['id']
        except TypeError:
            continue
        elem_rec = data['elements'][id]
        target = elem['boundary']
        temperatures.setdefault(target, 20)
        other_temperature = temperatures.get(target)
        room_temperature = temperatures.get(room_name)
        delta_t =  other_temperature - room_temperature
        elem_type = elem_rec['type']
        elem_area = elem_rec['A']

        elem_type_u = element_sliders[elem_type].value
        watts_per_kelvin = elem_type_u * elem_area
        flow_watts = delta_t * watts_per_kelvin 
        room_tot_flow_watts += flow_watts
    return room_tot_flow_watts

def calculate_target_temperature(t, room_name_alias):
    target_t_lagged = target_t = setpoint_slider.value - (night_set_back_slider.value if (t.hour < 6 or t.hour <= 23) else 0 )

    if real_setpoints_switch.active:
        if room_name_alias in house_data.room_setpoints:
            realset = max(house_data.room_setpoints[room_name_alias].get(t, 21) + setpoint_delta_slider.value, setpoint_minimum_slider.value)

            if realset is not None:
                target_t = realset
            realset_lagged = max(house_data.room_setpoints[room_name_alias].get(t-timedelta(minutes=radiator_response_time_slider.value), setpoint_slider.value) + setpoint_delta_slider.value, setpoint_minimum_slider.value)
            if realset_lagged is not None:
                target_t_lagged = realset_lagged
    return target_t,target_t_lagged

def calculate_room_name_alias(room_name):
    room_name_alias = room_name
    if room_name_alias == 'Medal area' or room_name_alias == 'Downstairs toilet' or room_name_alias == 'Dining room': 
        room_name_alias = 'Lounge'
    if room_name_alias == 'Guest suite shower room':
        room_name_alias = 'Guest suite'
    return room_name_alias


def calculate_flow_temperature(temperatures, t, real_temperatures):
    if real_temperatures:
        flow_t_recorded = house_data.flow_temperatures.get(t)
        flow_t = flow_temperature_slider.value if flow_t_recorded is None else flow_t_recorded + flow_temperature_reading_offset_slider.value
    else:
        if weather_compensation_switch.active:
            flow_t = flow_temperature_slider.value + weather_compensation_ratio_slider.value * max(0, weather_compensation_threshold_slider.value - temperatures['external']) - weather_compensation_ratio_slider.value / 2
        else:
            flow_t = flow_temperature_slider.value
    return flow_t

def scale_radiators():
    radiator_scales = {}
    total_base_rad_power = 0
    for room_name, room_data in data['rooms'].items():
        base_rad_power = sum([rad['heat50k'] for rad in room_data['radiators']])
        rad_density = max(100, base_rad_power) / room_data['area']
        scale = max(minimum_rad_density_slider.value, rad_density) / rad_density
        radiator_scales[room_name] = scale
        total_base_rad_power += base_rad_power
    return radiator_scales

def calculate_gas_efficiency(return_t):
    if return_t < 27: efficiency = 0.97
    elif return_t < 32: efficiency = 0.96
    elif return_t < 40: efficiency = 0.95
    elif return_t < 45: efficiency = 0.93
    elif return_t < 50: efficiency = 0.92
    elif return_t < 55: efficiency = 0.87
    else: efficiency = 0.86
    return efficiency


def update_data(attr, old, new):
    do_callback()

def do_update():
    room = room_keys[room_select.active]
    df = calculate_data(room)
    main_ds.data = df
    energy_use.text = work_out_energy_use(df)
    axs[1].title.text = f'{room} temperature'
    axs[3].title.text = f'{room} power discrepanices'

def work_out_energy_use(df):
    subset = df.filter(regex=r'.*_gain')
    subsetsum = subset.sum(axis=1)
    index_seconds = df.index.astype(np.int64) #  1e9
    #print(index_seconds.head())
    # heat_gain_ds.data =dict(x=df['time'], y=heat_gain_df[room])
    kwh_output =  scipy.integrate.trapezoid(subset.sum(axis=1), index_seconds) / 3.6e6
    kwh_input =  scipy.integrate.trapezoid(df['input_power'], index_seconds) / 3.6e6
    metered = sum(df['meters'])/1000
    return f'50th percentile power={subsetsum.quantile(0.5)/1e3:.1f}kW 90th percentile power={subsetsum.quantile(0.9)/1e3:.1f}kW 100th percentile power (max)={subsetsum.quantile(1.0)/1e3:.1f}kW total energy kwh output {kwh_output:.1f} metered {metered:.1f} input {kwh_input:.1f} efficency {100.0*kwh_output/ metered:.0f}%'

room_keys = sorted(data['rooms'].keys())
room_select = RadioGroup(labels=[str(x) for x in room_keys], active=0, inline=True)
room = room_keys[room_select.active]

power_slider =Slider(title='Heat source power', start=2000, end=40000, value=40000)
air_factor_slider = Slider(title='Air infiltation factor', start=0, end=10, value=0.33, step=0.01)
flow_temperature_slider = Slider(title='Flow temperature (C)', start=25, end=65, value=45)
t0 = isoparse("2023-08-01T00:00:00Z")
t1 = datetime.now()
t0p = isoparse("2024-02-01T00:00:00Z")
t1p = isoparse("2024-02-01T23:59:00Z")
day_range_slider = DateRangeSlider(width=800, start=t0, end=t1, value=(t0p,t1p))
minimum_rad_density_slider = Slider(title='Minimum rad density', start=30, end=1000, value=5)
weather_compensation_threshold_slider = Slider(title='Weather compensation threshold temperature', start=0, end=30, value=15, step=0.1)
weather_compensation_ratio_slider = Slider(title='Weather compensation ratio', start=0.1, end=1.5, value=0.6, step=0.05)
radiator_response_time_slider = Slider(title='Radiator response time', start=0, end=60, value=interval_minutes*2, step=interval_minutes)
flow_temperature_reading_offset_slider = Slider(title='Correction factor for flow temperature readings', start=-20, end=50, value=5)
temperature_change_factor_slider = Slider(title='Temperatue change ratio', start=1, end=1000, value=10)
setpoint_slider = Slider(title='Temperature setpoint', start=15, end=25, value=19, step=0.1)
night_set_back_slider = Slider(title="night set back", start=0, end=15, value=0)
setpoint_delta_slider = Slider(title="setpoint delta", start=-10, end=10, value=0, step=0.1)
setpoint_minimum_slider = Slider(title="setpoint minimum", start=0, end=21, value=10, step=0.5)
sliders = [ 
    day_range_slider, 
    power_slider, 
    air_factor_slider, 
    flow_temperature_slider, 
    minimum_rad_density_slider, 
    weather_compensation_threshold_slider, 
    weather_compensation_ratio_slider,
    night_set_back_slider,
    radiator_response_time_slider,
    flow_temperature_reading_offset_slider,
    temperature_change_factor_slider,
    setpoint_slider,
    setpoint_delta_slider,
    setpoint_minimum_slider
]
element_sliders = {}
for material in sorted(data['element_type'].keys()):
    element_sliders[material] = Slider(title=f"U value for {material}", start=0, end=3, step=0.05, value=data['element_type'][material]['uvalue'])
    sliders.append(element_sliders[material])
air_change_sliders = {}
for room in room_keys:
    air_change_sliders[room] = Slider(title=f'Air changes/h {room}', start=0, end=5, step=0.05, value=data['rooms'][room]['air_change_an_hour'] )
    sliders.append(air_change_sliders[room])

real_setpoints_switch = Switch(active=True)
weather_compensation_switch = Switch(active=False)
width = Span(dimension="width", line_dash="dashed", line_width=2)
height = Span(dimension="height", line_dash="dotted", line_width=2)
df = calculate_data(room, real_temperatures=True)
energy_text = work_out_energy_use(df)
room_colours = plasma(len(data['rooms']))
axs = []

TOOLTIPS = [("(x,y)", "(@time_str, $y)")]
for i in range(8):
    s = figure(height=400, width=800, x_axis_type='datetime', tools='hover,xwheel_zoom', tooltips=TOOLTIPS)
    s.add_tools(CrosshairTool(overlay=[width, height]))
    axs.append(s)

main_ds = ColumnDataSource(df)
axs[0].varea_stack(x= 'time', stackers=[x+'_gain' for x in data['rooms'].keys()],  source=main_ds, color=room_colours)
axs[0].scatter(x='time', y='meters', source=main_ds, color='black')

#axs[0].legend.location = 'bottom_right'
axs[0].legend.background_fill_alpha = 0.5
axs[0].title = 'Heat input, watts'

colours = {'external':'blue'}
room = list(data['rooms'].keys())[room_select.active]
col = room_colours[room_select.active]
axs[1].line(x='time', y='temperature',  source= main_ds,  line_width=2, color='blue')
axs[1].line(x='time', y='simulated_temperature',  source= main_ds,  line_width=2, color='lightgreen')

axs[1].line(x='time', y='setpoint',  source=main_ds,  line_width=2, color='red')
axs[1].line(x='time', y='setpoint_lagged',  source=main_ds,  line_width=2, color='pink')
axs[3].title = f'{room} power discrepanices'
#axs[i+3].y_range = Range1d(10, 25)
#axs[i+3].extra_y_ranges = {"power":Range1d(start=-2000, end=2000)}
#axs[i+3].line(x=df['time'], y=df[room], color='black')
colors = linear_cmap(field_name='temperature', palette='Viridis256', low=-5, high=30)
axs[3].scatter(x='time', y='power_discrepancy',  color=colors, source=main_ds)
#axs[i+3].add_layout(LinearAxis(y_range_name='power'), 'right')
axs[0].yaxis.axis_label = "Watts"
axs[1].legend.location = 'bottom_right'
axs[1].legend.background_fill_alpha = 0.5
axs[1].yaxis.axis_label = 'Celsius'
axs[1].title = f'{room} temperature'
axs[1].legend.click_policy = 'mute'

axs[2].title = 'Outside temperature'
axs[2].yaxis.axis_label = 'Celsius'
axs[2].line(x='time', y='external_temperature', source=main_ds)

axs[5].yaxis.axis_label = 'Power'
axs[5].title = 'Room heat loss'
axs[5].line(x='time', y='loss_for_room', source=main_ds)

axs[4].yaxis.axis_label = 'Power'
axs[4].title = 'Room heat gain'
axs[4].line(x='time', y='gain_for_room_unbounded', source=main_ds, line_color='yellow')
axs[4].line(x='time', y='gain_for_room', source=main_ds)
axs[4].line(x='time', y='available_radiator_power', source=main_ds, line_color='grey', alpha=0.4)


axs[6].title = 'Flow temperature'
axs[6].yaxis.axis_label = 'Celsius'
axs[6].line(x='time', y='flow_temperature', source=main_ds)

axs[7].title = 'Satisfaction'
axs[7].yaxis.axis_label = 'Ratio'
axs[7].line(x='time', y='satisfactions', source=main_ds)
def change_room(attr, old, new):
    do_callback()

def heat_pump_mode_callback():
    weather_compensation_switch.active = True
    power_slider.value = 12000
    real_setpoints_switch.active = False
    minimum_rad_density_slider.value = 300
    do_callback()

def boiler_mode_callback():
    weather_compensation_switch.active = False
    power_slider.value = 40000
    real_setpoints_switch.active = True
    minimum_rad_density_slider.value = 5
    do_callback()

def go_full_year():
    day_range_slider.start = t0
    day_range_slider.end = t1
    do_callback()

room_select.on_change('active', change_room)
energy_use = Div(text=energy_text)
boiler_button = Button(label='Switch to boiler settings (default)')
boiler_button.on_click(boiler_mode_callback)
heat_pump_mode_button = Button(label='Switch to heat pump settings')
heat_pump_mode_button.on_click(heat_pump_mode_callback)
full_year_button = Button(label='Switch to full year')
full_year_button.on_click(go_full_year)
layout = column([row([room_select])]+row_split(sliders, 4)+
    [row([
        Div(text='Use historical setpoints'), real_setpoints_switch,
        boiler_button,
        heat_pump_mode_button,
        full_year_button
    ]),
    row([energy_use]),
    row([Div(text='Weather compensation'), weather_compensation_switch])]+     
    row_split(axs, 2)
)
curdoc().add_root(layout)
    
#curdoc().add_periodic_callback(update_data, 10)
idle_callbacks = []

def do_callback():
    while idle_callbacks:
        try:
            curdoc().remove_timeout_callback(idle_callbacks.pop())
        except ValueError:
            pass
    idle_callbacks.append( curdoc().add_timeout_callback(do_update, 500))

for slider in sliders:
    slider.on_change('value', update_data)
for switch in [weather_compensation_switch, real_setpoints_switch]:
    switch.on_change('active', update_data)

if __name__ == '__main__':
    output_file('heatmodel.html')
    show(layout)
