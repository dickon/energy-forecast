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
class InfluxConfig(NamedTuple):
    token: str
    server: str
    bucket: str
    org: str

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

def calculate_data(room: str='') -> pd.DataFrame:
    out_temperatures = []
    temperatures = {}
    start_t = t = pytz.utc.localize(datetime.fromtimestamp(day_range_slider.value[0]/1000))
    end_t = pytz.utc.localize(datetime.fromtimestamp(day_range_slider.value[-1]/1000))
    print('start', start_t, "end", end_t)
    radiator_scales = {}
    total_base_rad_power = 0
    for room_name, room_data in data['rooms'].items():
        base_rad_power = sum([rad['heat50k'] for rad in room_data['radiators']])
        rad_density = max(100, base_rad_power) / room_data['area']
        scale = max(minimum_rad_density_slider.value, rad_density) / rad_density
        radiator_scales[room_name] = scale
        total_base_rad_power += base_rad_power
    print(f'Total base radiator {total_base_rad_power/1e3}kW')
    
    #pprint.pprint(radiator_scales)
    cursor = 0
    powers = []
    satisfactions = []
    system_power = power_slider.value
    input_power_series = []
    recs = { "time":[], "output_power":powers, "input_power":input_power_series,  "satisfactions":satisfactions, "meters":[]}

    while t < end_t:
        while cursor+ 1< len(house_data.outside_temperatures) and house_data.outside_temperatures[cursor+1].time < t:
            cursor += 1
        next_t = t + timedelta(minutes=interval_minutes)
        real_send_t = house_data.flow_temperatures.get(t)
        if real_send_t is None:
            real_send_t = flow_temperature_slider.value
        else:
            real_send_t = real_send_t + flow_temperature_reading_offset_slider.value
        recs.setdefault('send_temperature', []).append(real_send_t)
        rec = house_data.outside_temperatures[cursor]
        gas_reading = house_data.gas_readings.get(t, 0.0)
        recs['meters'].append(gas_reading)
        temperatures['external'] = rec.temperature
        recs['time'].append(t)
        satisfaction = 1.0
        if real_temperatures_switch.active:
            flow_t = real_send_t
        elif weather_compensation_switch.active:
            flow_t = flow_temperature_slider.value + weather_compensation_ratio_slider.value * max(0, 17.0 - temperatures['external'])
            if False:
                print('outside', temperatures['external'], 'so using weather compensation flow temperature', flow_t)
        else:
            flow_t = flow_temperature_slider.value
        return_t = flow_t - 20.0
        if return_t < 27: efficiency = 0.97
        elif return_t < 32: efficiency = 0.96
        elif return_t < 40: efficiency = 0.95
        elif return_t < 45: efficiency = 0.93
        elif return_t < 50: efficiency = 0.92
        elif return_t < 55: efficiency = 0.87
        else: efficiency = 0.86

        weather_compensation_ratio = 1.0
        room_powers = {}
        discrepanices = {}
        for phase in [0,1]:
            house_rad_output_watts = 0
            for room_index, (room_name, room_data) in enumerate(data['rooms'].items()):
                target_t_lagged = target_t = (21.5 if room_name != 'Master suite' else 19)
                room_name_alias = room_name
                if room_name_alias == 'Medal area' or room_name_alias == 'Downstairs toilet' or room_name_alias == 'Dining room': 
                    room_name_alias = 'Lounge'
                if room_name_alias == 'Guest suite shower room':
                    room_name_alias = 'Guest suite'

                if real_setpoints_switch.active:
                    if room_name_alias in house_data.room_setpoints:
                        realset = house_data.room_setpoints[room_name_alias].get(t)
                        if realset is not None:
                            target_t = realset
                        realset_lagged = house_data.room_setpoints[room_name_alias].get(t-timedelta(minutes=radiator_response_time_slider.value))
                        if realset_lagged is not None:
                            target_t_lagged = realset_lagged
                if phase == 1:
                    recs.setdefault(room_name+"_setpoint", []).append(target_t)

                verbose = room_name == 'Downstairs study'  and False
                #print(room_name)
                room_tot_flow_watts = 0
                temperatures.setdefault(room_name, 20)
                delta_t = temperatures['external'] - temperatures[room_name]

                infiltration_watts = room_data['air_change_an_hour'] * air_factor_slider.value * room_data['volume'] * delta_t
                if False:
                    print(f'{room_name} infiltration={infiltration_watts} from delta T {delta_t}')
                room_tot_flow_watts += infiltration_watts
                if not (t.hour >= 6 and t.hour < 23):
                    target_t_lagged -= night_set_back_slider.value
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
                    elem_type_rec = data['element_type'][elem_type]
                    elem_type_u = elem_type_rec['uvalue']
                    watts_per_kelvin = elem_type_u * elem_area
                    flow_watts = delta_t * watts_per_kelvin 
                    if False and room == 'Master suite' and phase == 1:
                        print(f'  {elem_rec["width"]:.02f}*{elem_rec["height"]:.02f} {elem_rec["type"]} WK={watts_per_kelvin:.1f} to {elem["boundary"]}@{other_temperature}dT{delta_t} flow={flow_watts:.0f}W')
                    room_tot_flow_watts += flow_watts
                if phase == 1:               
                    if False and room == 'Master suite':
                        print(f' total {room_tot_flow_watts:.0f}W')         
                    recs.setdefault(f'{room_name}_loss', []).append(-room_tot_flow_watts)
                available_rad_watts = 0
                for rad in room_data['radiators']:
                    temperatures.setdefault(room_name, 21)
                    rad_delta_t = max(0, flow_t - temperatures[room_name])
                    rad_watts = (rad['heat50k'] *radiator_scales[room_name] * rad_delta_t / 50) * satisfaction 
                    if False and room_name == room:
                        print(f'  { room_name} rad {rad["name"]} 50K {rad['heat50k']} rad delta T {rad_delta_t} scale { radiator_scales[room_name] } satisfaction { satisfaction } power {rad_watts:.0f}W')
                    available_rad_watts += rad_watts
                if temperatures[room_name] < target_t_lagged+0.5:
                    # room too cold; rads run flat out
                    room_rad_output= available_rad_watts
                elif temperatures[room_name] < target_t_lagged+1.5:
                    # room about right; rads run at heat output
                    room_rad_output = -room_tot_flow_watts
                else:
                    # room too hot; rads off
                    room_rad_output = 0
                room_tot_flow_watts += room_rad_output
                house_rad_output_watts += room_rad_output

                if phase == 1:
                    recs.setdefault(room_name+'_gain', []).append(room_rad_output)
                    temp_change = room_tot_flow_watts * interval_minutes / 60 / (room_data['volume']*temperature_change_factor_slider.value)
                    orig_temp = temperatures[room_name]
                    temperatures[room_name] += temp_change
                    if real_temperatures_switch.active and room_name_alias in house_data.room_temperatures:
                        realtemp = house_data.room_temperatures[room_name_alias].get(t)
                        if realtemp is not None:
                            #print('real temp', realtemp, 'for',room_name, 'at', next_t, 'c/w caluclated', temperatures[room_name])
                            actual_temp_change = realtemp - orig_temp
                            actual_flow = actual_temp_change * (room_data['volume']*temperature_change_factor_slider.value)
                            #print('actual flow', actual_flow, 'calculated', room_tot_flow, 'for', room_name, 'at', next_t)
                            delta = actual_flow - room_tot_flow_watts
                            discrepanices[room_name] = delta
                            temperatures[room_name] = realtemp
                    else:
                        #print('room missing for', room_name)
                        pass

                    if verbose:
                        print(f'   Total room flow={room_tot_flow_watts:.0f}W area {room_data['area']:.0f}qm^2 temperature +{temp_change}->{temperatures[room_name]}')
            if phase == 0:
                if house_rad_output_watts == 0: house_rad_output_watts = 500
                # if we need 20 kw and system power is 10kw then satisfaction = 0.5
                # if we need 5 kw and system power is 10kw then satisfaction = 1.0
                ideal_power_out = house_rad_output_watts
                if ideal_power_out < system_power:
                    satisfaction = 1.0
                else:
                    satisfaction =  system_power / ideal_power_out
                satisfactions.append(satisfaction)
            if phase == 1:
                input_power_series.append ( house_rad_output_watts / efficiency)
                if False:
                    print(f'{t} power {house_rad_output_watts/1e3:.1f}kW (ideal:{ideal_power_out/1e3:.1f}kW) inside={temperatures['Downstairs study']:.1f}C outside={temperatures['external']:.1f}C power={house_rad_output_watts} satisfaction={satisfaction*100:.0f}%')

        t = next_t
        for k,v  in temperatures.items():
            recs.setdefault(k+'_temperature', []).append(v)
        for k in data['rooms'].keys():
            recs.setdefault(k+'_power_error', []).append(discrepanices.get(k, 0))
        assert house_rad_output_watts >= 0
        powers.append(house_rad_output_watts)

    df = pd.DataFrame(recs)
    df['loss_for_room'] = df[room+'_loss']
    df['gain_for_room'] = df[room+'_gain']
    df['power_error'] = df[room+'_power_error']
    df['temperature'] = df[room+'_temperature']
    df['setpoint'] = df[room+'_setpoint']
    return df


def update_data(attr, old, new):
    do_callback()

def do_update():
    room = list(data['rooms'].keys())[room_select.active]
    df = calculate_data(room)
    main_ds.data = df
    energy_use.text = work_out_energy_use(df)
    axs[1].title.text = f'{room} temperature'
    axs[3].title.text = f'{room} power discrepanices'

def work_out_energy_use(df):
    subset = df.filter(regex=r'.*_gain')
    print(subset)
    subsetsum = subset.sum(axis=1)
    index_seconds = df['time'].astype(np.int64) // 1e9
    #print(index_seconds.head())
    # heat_gain_ds.data =dict(x=df['time'], y=heat_gain_df[room])
    kwh_output =  scipy.integrate.trapezoid(subset.sum(axis=1), index_seconds) / 3.6e6
    kwh_input =  scipy.integrate.trapezoid(df['input_power'], index_seconds) / 3.6e6
    metered = sum(df['meters'])/1000
    return f'50th percentile power={subsetsum.quantile(0.5)/1e3:.1f}kW 90th percentile power={subsetsum.quantile(0.9)/1e3:.1f}kW 100th percentile power (max)={subsetsum.quantile(1.0)/1e3:.1f}kW total energy kwh output {kwh_output:.1f} metered {metered:.1f} input {kwh_input:.1f} efficency {100.0*kwh_output/ metered:.0f}%'

room_select = RadioGroup(labels=[str(x) for x in data['rooms'].keys()], active=10, inline=True)
room = list(data['rooms'].keys())[room_select.active]
    
power_slider =Slider(title='Heat source power', start=2000, end=40000, value=40000)
air_factor_slider = Slider(title='Air infiltation factor', start=0, end=10, value=0.33, step=0.01)
flow_temperature_slider = Slider(title='Flow temperature (C)', start=25, end=65, value=56)
t0 = isoparse("2023-08-01T00:00:00Z")
t1 = datetime.now()
t0p = isoparse("2024-02-01T00:00:00Z")
t1p = isoparse("2024-02-03T23:59:00Z")
day_range_slider = DateRangeSlider(width=800, start=t0, end=t1, value=(t0p,t1p))
minimum_rad_density_slider = Slider(title='Minimum rad density', start=30, end=1000, value=5)
weather_compensation_ratio_slider = Slider(title='Weather compensation ratio', start=0.1, end=1.5, value=0.6, step=0.05)
radiator_response_time_slider = Slider(title='Radiator response time', start=0, end=60, value=interval_minutes*2, step=interval_minutes)
flow_temperature_reading_offset_slider = Slider(title='Correction factor for flow temperature readings', start=-20, end=50, value=5)
temperature_change_factor_slider = Slider(title='Temperatue change ratio', start=1, end=1000, value=10)
night_set_back_slider = Slider(title="night set back", start=0, end=15, value=0)
sliders = [ day_range_slider, power_slider, air_factor_slider, flow_temperature_slider, minimum_rad_density_slider, weather_compensation_ratio_slider, night_set_back_slider, radiator_response_time_slider, flow_temperature_reading_offset_slider, temperature_change_factor_slider]
real_temperatures_switch = Switch(active=True)
real_setpoints_switch = Switch(active=True)
weather_compensation_switch = Switch(active=False)
width = Span(dimension="width", line_dash="dashed", line_width=2)
height = Span(dimension="height", line_dash="dotted", line_width=2)
df = calculate_data(room)
energy_text = work_out_energy_use(df)
room_colours = plasma(len(data['rooms']))
axs = []

for i in range(7):
    s = figure(height=400, width=800, x_axis_type='datetime', tools='hover,xwheel_zoom')
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
axs[1].line(x='time', y='setpoint',  source=main_ds,  line_width=2, color='red')
axs[3].title = f'{room} power discrepanices'
#axs[i+3].y_range = Range1d(10, 25)
#axs[i+3].extra_y_ranges = {"power":Range1d(start=-2000, end=2000)}
#axs[i+3].line(x=df['time'], y=df[room], color='black')
colors = linear_cmap(field_name='temperature', palette='Viridis256', low=-5, high=30)
axs[3].scatter(x='time', y='power_error',  color=colors, source=main_ds)
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
axs[4].line(x='time', y='gain_for_room', source=main_ds)

axs[6].title = 'Flow temperature'
axs[6].yaxis.axis_label = 'Celsius'
axs[6].line(x='time', y='send_temperature', source=main_ds)
def change_room(attr, old, new):
    do_callback()

def heat_pump_mode_callback():
    weather_compensation_switch.active = True
    power_slider.value = 12000
    real_temperatures_switch.active = False
    real_setpoints_switch.active = False
    do_callback()

def boiler_mode_callback():
    weather_compensation_switch.active = False
    power_slider.value = 40000
    real_temperatures_switch.active = True
    real_setpoints_switch.active = True
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
layout = column([
    row([room_select]), 
    row(sliders[:3]), 
    row(sliders[3:7]), 
    row(sliders[7:]), 
    row([
        Div(text='Use historical temperatures'), real_temperatures_switch,
        Div(text='Use historical setpoints'), real_setpoints_switch,
        boiler_button,
        heat_pump_mode_button,
        full_year_button
    ]),
    row([energy_use]),
    row([Div(text='Weather compensation'), weather_compensation_switch]),     
    row(axs[:2]),
    row(axs[2:4]),
    row(axs[4:6]),
    row(axs[6:8])])
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
for switch in [weather_compensation_switch, real_temperatures_switch, real_setpoints_switch]:
    switch.on_change('active', update_data)

if __name__ == '__main__':
    output_file('heatmodel.html')
    show(layout)
