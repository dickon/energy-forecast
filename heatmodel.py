import json, pprint, pickle, os
from dateutil.parser import isoparse
from datetime import datetime, timedelta
from typing import NamedTuple
import influxdb_client, time
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
import pandas as pd
from bokeh.layouts import column, row
from bokeh.plotting import figure, show
from bokeh.models import Slider, Switch, Div, CrosshairTool, Span, HoverTool, ColumnDataSource
from bokeh.io import curdoc
from structures import HouseData, TemperaturePoint
from bokeh.palettes import magma
import scipy.integrate


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
influxConfig = InfluxConfig(**site["influx"])
client = influxdb_client.InfluxDBClient(
    url=influxConfig.server, token=influxConfig.token, org=influxConfig.org
)

write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()

def fetch_house_data() -> HouseData:
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
  |> aggregateWindow(every: 1h, fn: mean, createEmpty: false)
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
    return HouseData(outside_temperatures=temps, room_setpoints=room_setpoints, room_temperatures=room_temperatures)


def load_house_data() -> HouseData:
    CACHE = 'house_data.pickle'
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
print(house_data.room_temperatures.keys())

with open('heatmodel.json', 'r') as f:
    data = json.load(f)

def calculate_data():
    timestamps = []
    out_temperatures = []
    power_errors = []
    temperatures = {}
    start_t = t = house_data.outside_temperatures[0].time.replace(second=0, minute=0, microsecond=0)
    end_t = start_t + timedelta(days=day_range_slider.value)
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
    room_powers_series = {k: [] for k in data['rooms'].keys()}
    input_power_series = []
    while t < end_t:
        while house_data.outside_temperatures[cursor+1].time < t:
            cursor += 1
        next_t = t + timedelta(hours=1)
        rec = house_data.outside_temperatures[cursor]
        temperatures['external'] = rec.temperature
        timestamps.append(t)
        satisfaction = 1.0
        if weather_compensation_switch.active:
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
            house_rad_output = 0
            for room_index, (room_name, room_data) in enumerate(data['rooms'].items()):
                target_t = (21.5 if room_name != 'Master suite' else 19) 
                room_name_alias = room_name
                if room_name_alias == 'Medal area' or room_name_alias == 'Downstairs toilet' or room_name_alias == 'Dining room': 
                    room_name_alias = 'Lounge'
                if room_name_alias == 'Guest suite shower room':
                    room_name_alias = 'Guest suite'

                if real_temperatures_switch.active:
                    if room_name_alias in house_data.room_setpoints:
                        realset = house_data.room_setpoints[room_name_alias].get(t)
                        if realset is not None:
                            #print('override setpoint', room_name, 'at', t, 'from', target_t, 'to', realset)
                            target_t = realset
                        else:
                            #print('override missing for', room_name, 'at', t)
                            pass
                    else:
                        #print('no overrides for', room_name_alias)
                        pass
                verbose = room_name == 'Downstairs study'  and False
                #print(room_name)
                room_tot_flow = 0
                
                if not (t.hour >= 6 and t.hour < 23):
                    target_t -= night_set_back_slider.value
                for elem in room_data['elements']:
                    try:
                        id = elem['id']
                    except TypeError:
                        continue
                    elem_rec = data['elements'][id]
                    target = elem['boundary']
                    temperatures.setdefault(target, 20)
                    temperatures.setdefault(room_name, 20)
                    other_temperature = temperatures.get(target)
                    room_temperature = temperatures.get(room_name)
                    delta_t =  other_temperature - room_temperature
                    flow = delta_t * elem_rec['wk']
                    #print(f'  {elem_rec["width"]:.02f}*{elem_rec["height"]:.02f} {elem_rec["type"]} WK={elem_rec["wk"]:.1f} to {elem["boundary"]}@{other_temperature}dT{delta_t} {flow:.0f}W')
                    room_tot_flow += flow
                room_rad_output = 0
                for rad in room_data['radiators']:
                    temperatures.setdefault(room_name, 21)
                    rad_delta_t = max(0, flow_t - temperatures[room_name])
                    rad_power = (rad['heat50k'] *radiator_scales[room_name] * rad_delta_t / 50 if temperatures[room_name] < target_t else 0) * satisfaction 
                    if verbose:
                        print(f'  { room_name} rad {rad["name"]} rad delta T {rad_delta_t} power {rad_power:.0f}W')
                    room_tot_flow += rad_power
                    room_rad_output += rad_power
                house_rad_output += room_rad_output
                if phase == 1:
                    room_powers_series[room_name].append(room_rad_output)
                    temp_change = room_tot_flow / (room_data['area']*300)
                    orig_temp = temperatures[room_name]
                    temperatures[room_name] += temp_change
                    if room_name_alias in house_data.room_temperatures:
                        realtemp = house_data.room_temperatures[room_name_alias].get(next_t)
                        if realtemp is not None:
                            #print('real temp', realtemp, 'for',room_name, 'at', next_t, 'c/w caluclated', temperatures[room_name])
                            actual_temp_change = realtemp - orig_temp
                            actual_flow = actual_temp_change * (room_data['area']*300)
                            #print('actual flow', actual_flow, 'calculated', room_tot_flow, 'for', room_name, 'at', next_t)
                            discrepanices[room_name] = actual_flow - room_tot_flow
                            temperatures[room_name] = realtemp
                            
                        else:
                            #print('temp missing for', room_name, 'at', t)
                            #print('have', house_data.room_temperatures[room_name].keys())
                            pass
                    else:
                        #print('room missing for', room_name)
                        pass

                    if verbose:
                        print(f'   Total room flow={room_tot_flow:.0f}W area {room_data['area']:.0f}qm^2 temperature +{temp_change}->{temperatures[room_name]}')
            if phase == 0:
                if house_rad_output == 0: house_rad_output = 500
                # if we need 20 kw and system power is 10kw then satisfaction = 0.5
                # if we need 5 kw and system power is 10kw then satisfaction = 1.0
                ideal_power_out = house_rad_output
                if ideal_power_out < system_power:
                    satisfaction = 1.0
                else:
                    satisfaction =  system_power / ideal_power_out
                satisfactions.append(satisfaction)
            if phase == 1:
                input_power_series.append ( house_rad_output / efficiency)
                if False:
                    print(f'{t} power {house_rad_output/1e3:.1f}kW (ideal:{ideal_power_out/1e3:.1f}kW) inside={temperatures['Downstairs study']:.1f}C outside={temperatures['external']:.1f}C power={house_rad_output} satisfaction={satisfaction*100:.0f}%')

        t = next_t
        out_temperatures.append( dict(**temperatures))
        power_errors.append(dict(**discrepanices))
        assert house_rad_output >= 0
        powers.append(house_rad_output)
    recs = { "time":timestamps, "output_power":powers, "input_power":input_power_series,  "satisfactions":satisfactions}

    room_powers_df = pd.DataFrame( room_powers_series, index=timestamps )
    for k in temperatures:
        recs[k] = [x[k] for x in out_temperatures] 
    df = pd.DataFrame(recs)
    power_errors_df = pd.DataFrame(power_errors)
    return room_powers_df, df, power_errors_df

def update_data(attrname, old, new):
    room_powers_series, df, power_errors_df = calculate_data()

    # for ds, values in zip(ds_power, room_powers_series):
    #     ds.data = values
    for room in data['rooms']:
        ds_rooms[room].data = dict(x=df['time'], y=df[room],)
        print(f'room {room} mean temperature {df[room].mean()}')
    print(power_errors_df.to_string())
    room_powers_ds.data = ColumnDataSource.from_df(room_powers_series)
    ds_outside.data = dict(x=df['time'], y=df['external'])
    subset = room_powers_series
    subsetsum = subset.sum(axis=1)
    print('50% percentile power', subsetsum.quantile(0.5))
    print('90% percentile power', subsetsum.quantile(0.9))
    print('100% percentile power', subsetsum.quantile(1.0))
    index_seconds = room_powers_series.index.astype(np.int64) // 1e9
    #print(index_seconds.head())

    kwh_output =  scipy.integrate.trapezoid(subset.sum(axis=1), index_seconds) / 3.6e6
    kwh_input =  scipy.integrate.trapezoid(df['input_power'], index_seconds) / 3.6e6
    print(f'total energy kwh output {kwh_output:.1f} input {kwh_input:.1f} efficency {100.0*kwh_output/ kwh_input:.0f}%')

power_slider =Slider(title='Heat source power', start=2000, end=40000, value=40000)
flow_temperature_slider = Slider(title='Flow temperature (C)', start=25, end=65, value=50)
day_range_slider = Slider(title='Days to model', start=10, end=365, value=365)
minimum_rad_density_slider = Slider(title='Minimum rad density', start=30, end=1000, value=5)
weather_compensation_ratio_slider = Slider(title='Weather compensation ratio', start=0.1, end=1.5, value=0.6, step=0.05)
night_set_back_slider = Slider(title="night set back", start=0, end=15, value=0)
sliders = [power_slider, flow_temperature_slider, day_range_slider, minimum_rad_density_slider, weather_compensation_ratio_slider, night_set_back_slider]
real_temperatures_switch = Switch(active=True)
weather_compensation_switch = Switch(active=False)
width = Span(dimension="width", line_dash="dashed", line_width=2)
height = Span(dimension="height", line_dash="dotted", line_width=2)
room_powers_series, df, power_errors_df = calculate_data()
room_colours = magma(len(data['rooms']))
axs = []
for i in range(3):
    #h = HoverTool(  )
    s = figure(height=300, width=800, x_axis_type='datetime', tools='hover,xwheel_zoom')
    s.add_tools(CrosshairTool(overlay=[width, height]))
    s.sizing_mode = 'scale_width'
    axs.append(s)

print(room_powers_series)
room_powers_ds = ColumnDataSource(room_powers_series)
axs[0].varea_stack(stackers=data['rooms'].keys(), x= 'index', source=room_powers_ds, color=room_colours)
#axs[0].legend.location = 'bottom_right'
axs[0].legend.background_fill_alpha = 0.5


#ds_power = [x.data_source for x in power_stack]
#axs[0].varea_stack(stackers=room_powers_series)
axs[0].title = 'Heat input, watts'

colours = {'external':'blue'}
ds_rooms = {}
for room, col in zip(data['rooms'], room_colours):

    r = axs[1].line(x=df['time'], y=df[room], legend_label=room, line_width=2, color=colours.get(room, col), muted_alpha=0.2, alpha=0.9)
    ds_rooms[room] = r.data_source
    if room != 'Lounge':
        r.muted = True
axs[0].yaxis.axis_label = "Watts"
axs[1].legend.location = 'bottom_right'
axs[1].legend.background_fill_alpha = 0.5
axs[1].yaxis.axis_label = 'Celsius'
axs[1].title = 'Room temperatures'
axs[1].legend.click_policy = 'mute'
axs[2].title = 'Outside temperature'
axs[2].yaxis.axis_label = 'Celsius'
ds_outside = axs[2].line(x=[],y=[]).data_source

curdoc().add_root(column(row(*sliders[:4]), row(Div(text='Use historical real temperatures'), real_temperatures_switch, Div(text='Weather compensation'), weather_compensation_switch, *sliders[4:]), axs[0], axs[1], axs[2]))
update_data(None, None, None)
for slider in sliders:
    slider.on_change('value', update_data)
for switch in [weather_compensation_switch, real_temperatures_switch]:
    switch.on_change('active', update_data)