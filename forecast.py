import datetime
import json
import matplotlib
import os.path
import pprint
import pysolar
import random
import traceback
import influxdb_client
import matplotlib.pyplot as plt
from influxdb_client.client.write_api import SYNCHRONOUS

site = json.load(open(os.path.expanduser("~/src/powerwallextract/site.json")))

t0 = datetime.datetime.strptime(
    "2023-01-01 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z"
)  # start of modelling period
t1 = datetime.datetime.strptime(
    "2023-12-31 23:59:00 Z", "%Y-%m-%d %H:%M:%S %z"
)  # end of modelling period
tnow = datetime.datetime.strptime(
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S Z"), "%Y-%m-%d %H:%M:%S %z"
)

URL = site["influx"]["server"]

client = influxdb_client.InfluxDBClient(
    url=URL, token=site["influx"]["token"], org=site["influx"]["org"]
)
write_api = client.write_api(write_options=SYNCHRONOUS)
query_api = client.query_api()
time_string_format = "%Y-%m-%dT%H:%M:%SZ"


def time_string(dt):
    return dt.strftime(time_string_format)


def do_query(
    query: str,
    bucket: str,
    t0: datetime.datetime,
    t1: datetime.datetime,
    extra: str = "",
    verbose=False,
    multi=True,
):
    import time

    fquery = f"""
            {extra}
            from(bucket: "{bucket}") |> range(start: {time_string(t0)}, stop: {time_string(t1)}) 
            {query}
"""
    start = time.time()
    if verbose:
        print(fquery)
    resl = query_api.query(org=site["influx"]["org"], query=fquery)
    end = time.time()
    delay = end - start

    if len(resl) == 0:
        if verbose:
            print("no results in", delay)
        return []
    if verbose:
        print("results in", delay)
    if multi:
        l = []
        for s in resl:
            for i in s:
                l.append(i)
        return l
    return resl[0]


def json_cache(filename, generate, max_age_days=7):
    if os.path.exists(filename):
        mtime = os.stat(filename).st_mtime
        mtime_date = datetime.datetime.fromtimestamp(mtime)
        age = datetime.datetime.now() - mtime_date
        print("cache file", filename, "age", age)
        if age.days < max_age_days:
            try:
                with open(filename, "r") as fp:
                    return json.load(fp)
            except:
                print(f"unable read {filename}; generating")
                traceback.print_exc()
    data = generate()
    with open(filename, "w") as fp:
        json.dump(data, fp, indent=2)
    return data


def generate_mean_time_of_day():
    usage_actual = {}
    txover = datetime.datetime.strptime("2023-07-30 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z")
    t = t0

    def query_usage(t):
        end_of_hh = t + datetime.timedelta(minutes=29)
        if (
            t.month == 8 and (t.day > 12 and t.day < 24) or t.day == 8 or t.day == 7
        ) or (t.month == 7 and t.day in range(27, 32)):
            return None, False
        if t < txover:
            res = do_query(
                """|> filter(fn: (r) => r["_measurement"] == "energy_usage")
            |> filter(fn: (r) => r["_field"] == "usage")
            |> filter(fn: (r) => r["account_name"] == "Primary Residence")
            |> filter(fn: (r) => r["detailed"] == "False")
            |> filter(fn: (r) => r["device_name"] == "Home-TotalUsage")
            |> filter(fn: (r) => r._value > 0.0)
            |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
            |> yield(name: "last")""",
                "vue",
                t,
                end_of_hh,
            )
            abs = True
        else:
            res = do_query(
                """
            |> filter(fn: (r) => r["_measurement"] == "energy")
            |> filter(fn: (r) => r["_field"] == "energy_imported")
            |> filter(fn: (r) => r["meter"] == "load") |> last() """,
                "powerwall",
                t,
                end_of_hh,
            )
            abs = False
        if res == []:
            return None, abs
        data = [(x["_stop"], x["_value"]) for x in res]

        if (data[0][0] - t).seconds < 1800:
            return data[0][1], abs
        else:
            return None, abs

    time_of_day = {}
    base = None
    base_t = None

    while t < tnow:
        now, abs = query_usage(t)
        if abs:
            usage = now / 2 if now else 0
        else:
            if now:
                if base and t - base_t < datetime.timedelta(minutes=60):
                    usage = now - base
                else:
                    usage = None
                base = now
                base_t = t
            else:  #
                usage = None
        t = t + datetime.timedelta(minutes=30)
        if usage is not None and usage > 10:
            tday = f"{t.hour:02d}:{t.minute:02d}"
            time_of_day.setdefault(tday, list())
            time_of_day[tday].append(usage)
            usage_actual[t.isoformat()] = usage

    return {
        "mean_time_of_day": [(t, sum(x) / len(x)) for (t, x) in time_of_day.items()],
        "usage_actual": usage_actual,
    }


data = json_cache("kwh_use_time_of_day.json", generate_mean_time_of_day)
mean_time_of_day = data["mean_time_of_day"]
usage_actual_text = data["usage_actual"]
usage_actual = {
    datetime.datetime.fromisoformat(x[0]): x[1] for x in usage_actual_text.items()
}
values = [x for x in usage_actual.items()]
plt.figure(figsize=(12, 8))
plt.scatter(
    [t for t, _ in values],
    [t.hour + t.minute / 60 for t, _ in values],
    s=[v / 100 for _, v in values],
    c=[v / 100 for _, v in values],
)


import matplotlib.pyplot as plt
import math
import datetime

azi_resolution = 3
alt_resolution = 1


def get_solar_position_index(t):
    altitude = pysolar.solar.get_altitude(
        site["location"]["latitude"],
        site["location"]["longitude"],
        t + datetime.timedelta(minutes=15),
    )
    azimuth = pysolar.solar.get_azimuth(
        site["location"]["latitude"],
        site["location"]["longitude"],
        t + datetime.timedelta(minutes=15),
    )
    return alt_resolution * round(altitude / alt_resolution), azi_resolution * round(
        azimuth / azi_resolution
    )


txover = datetime.datetime.strptime(
    site["powerwall"]["commission_date"], "%Y-%m-%d %H:%M:%S %z"
)


def query_solar(t):
    res = do_query(
        """
    |> filter(fn: (r) => r["_measurement"] == "power")
        |> filter(fn: (r) => r["_field"] == "instant_power")
        |> filter(fn: (r) => r["meter"] == "solar")
                   |> mean()
 """,
        "powerwall",
        t,
        t + datetime.timedelta(minutes=30),
    )
    data = [(x["_stop"], x["_value"]) for x in res]
    if data:
        return data[0][1]


def generate_solar_model():
    bucket = "56new"

    solar_output_w = {}
    solar_pos_model = {}
    base = None
    t = tcommission = datetime.datetime.strptime(
        site["solar"]["commission_date"], "%Y-%m-%d %H:%M:%S %z"
    )
    # t = tnow - datetime.timedelta(hours=240)
    tbad0 = datetime.datetime.strptime("2023-08-11 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z")
    tbad1 = datetime.datetime.strptime("2023-08-19 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z")

    while t < tnow:
        usage = None
        if t > txover:
            usage = query_solar(t)

        else:
            resl = do_query(
                """
                |> filter(fn: (r) => r["_field"] == "usage")
                |> filter(fn: (r) => r["account_name"] == "Primary Residence")
                |> filter(fn: (r) => r["detailed"] == "True")
                |> filter(fn: (r) => r["device_name"] == "SolarEdge Inverter")
                |> mean() """,
                "vue",
                t,
                t + datetime.timedelta(minutes=29),
            )
            if resl:
                values = [-x["_value"] for x in resl]
                if values and values[0] > 10:
                    usage = values[0]
        if usage and (t < tbad0 or t > tbad1) and usage > 0:
            pos = get_solar_position_index(t)
            if pos[0] >= 0:
                solar_pos_model.setdefault(pos, list())
                solar_pos_model[pos].append(usage / 2)

            solar_output_w[t.isoformat()] = usage / 2

        t = t + datetime.timedelta(minutes=30)

    solar_model_table = dict()
    for pos, powl in solar_pos_model.items():
        solar_model_table[repr(pos)] = sum(powl) / len(powl)

    for azi in range(0, 360, azi_resolution):
        for alt in range(0, 70, alt_resolution):
            pos = (alt, azi)
            if pos not in solar_pos_model:
                mindist = None
                minv = None
                for altpos, powl in solar_pos_model.items():
                    dist = math.sqrt(
                        (altpos[0] - pos[0]) ** 2 * 10 + (altpos[1] - pos[1]) ** 2
                    )
                    if mindist is None or dist < mindist:
                        minv = powl
                        mindist = dist
                solar_pos_model[pos] = minv

    # backfill the gaps
    t = tcommission
    while t < tnow:
        t_iso = t.isoformat()
        pos = get_solar_position_index(t)
        if t_iso not in solar_output_w:
            if pos[0] < 0:
                solar_output_w[t_iso] = 0
            else:
                values = solar_pos_model[pos]
                solar_output_w[t_iso] = sum(values) / len(values)

        t = t + datetime.timedelta(minutes=30)

    record = {
        "table": solar_model_table,
        "output": solar_output_w,
        "model": [
            (
                repr(pos),
                sum(solar_pos_model[pos]) / len(solar_pos_model[pos]),
                len(solar_pos_model[pos]),
            )
            for pos in sorted(solar_pos_model.keys())
            if pos[0] >= 0
        ],
    }
    pprint.pprint(record)
    return record


solar_model_record = json_cache(
    "solar_model.json", generate_solar_model, max_age_days=7
)
solar_output_w = {
    datetime.datetime.fromisoformat(t): v
    for (t, v) in solar_model_record["output"].items()
}
solar_model = solar_model_record["model"]
solar_model_table = {eval(k): v for k, v in solar_model_record["table"].items()}


def plot_model(m):
    altitude = [x[0][0] for x in m]
    azimuth = [x[0][1] for x in m]
    pow = [x[1] * 2 for x in m]
    return plt.scatter(azimuth, altitude, [x / 60 for x in pow], pow)


def get_tariffs():
    rec = {}
    for outgoing in [True, False]:
        productq = (
            'r["product"] == "AGILE-OUTGOING-BB-23-02-28" or  r["product"] == "AGILE-OUTGOING-19-05-13"'
            if outgoing
            else 'r["product"] == "AGILE-FLEX-22-11-25" or r["product"] == "AGILE-BB-23-12-06"'
        )
        resl = do_query(
            f"""
      |> filter(fn: (r) => r["_measurement"] == "unit_cost")
      |> filter(fn: (r) => r["_field"] == "price")
      |> filter(fn: (r) => {productq} ) """,
            "tariffs",
            t0,
            t1,
            verbose=True,
            multi=True,
        )
        for res in resl:
            t = res["_time"]
            v = res["_value"]
            k = "agile " + ("outgoing" if outgoing else "incoming")
            rec.setdefault(k, dict())
            rec[k][t.isoformat()] = v
    gas_res = do_query(
        f"""
      |> filter(fn: (r) => r["_measurement"] == "unit_cost")
      |> filter(fn: (r) => r["fuel"] == "gas")
      |> filter(fn: (r) => r["product"] == "VAR-22-11-01")""",
        "tariffs",
        t0,
        t1,
        verbose=True)
    for res in gas_res:
        t = res["_time"]
        v = res["_value"]
        rec.setdefault('gas', dict())

        rec['gas'][t.date().strftime('%Y-%m-%d')] = v
    gas_sc = do_query(
    f"""
      |> filter(fn: (r) => r["_measurement"] == "standing_charge")
      |> filter(fn: (r) => r["fuel"] == "gas")
      |> filter(fn: (r) => r["product"] == "VAR-22-11-01")""",
    "tariffs",
    t0,
    t1,
    verbose=True)
    for res in gas_sc:
        t = res["_time"]
        v =  res["_value"]
        rec.setdefault('gas standing', dict())
        rec['gas standing'][t.date().strftime('%Y-%m-%d')] = v

    return rec


tariffs = json_cache("tariffs.json", get_tariffs)
agile_incoming_cache = {}
gas_tariffs_cache = {}
for outgoing in [True, False]:
    k = "agile " + ("outgoing" if outgoing else "incoming")
    for t, value in tariffs[k].items():
        agile_incoming_cache[(datetime.datetime.fromisoformat(t), outgoing)] = value
for t, value in tariffs['gas'].items():
    gas_tariffs_cache[(True, datetime.datetime.fromisoformat(t))] = value
for t, value in tariffs['gas standing'].items():
    gas_tariffs_cache[(False, datetime.datetime.fromisoformat(t))] = value

plt.figure(figsize=(12, 8))

p = plot_model([(pos, v) for pos, v in solar_model_table.items()])
plt.xlabel("azimuth")
plt.ylabel("altitude")
plt.colorbar()

values = [x for x in solar_output_w.items() if x[1] > 20]
plt.figure(figsize=(12, 8))
plt.scatter(
    [t for t, _ in values],
    [t.hour + t.minute / 60 for t, _ in values],
    s=[v / 100 for _, v in values],
    c=[v * 2 for _, v in values],
)
plt.xlabel("date")
plt.ylabel("time of day")
plt.colorbar()


def get_daily_gas_use():
    tback = t0
    res = do_query(
        """
  |> filter(fn: (r) => r["_measurement"] == "usage")
  |> filter(fn: (r) => r["_field"] == "usage")
  |> filter(fn: (r) => r["device_name"] == "utility meter")
  |> filter(fn: (r) => r["energy"] == "gas" )
  |> window(every: 1d)
  |> sum()
  |> duplicate(column: "_start", as: "_time")
  |> map(fn: (r) => ({ r with _time: date.add(d: 12h, to: r._time) }))
  |> window(every: inf)
  """,
        "56new",
        tback,
        t1,
        """import "date" """,
    )
    res2 = do_query(
        """
  |> filter(fn: (r) => r["_measurement"] == "consumed")
  |> filter(fn: (r) => r["_field"] == "usage")
  |> filter(fn: (r) => r["device_name"] == "utility meter")
  |> filter(fn: (r) => r["energy"] == "gas")
    |> aggregateWindow(every: 24h, fn: mean,  createEmpty: false)
  |> map(fn: (r) => ({ r with _value: r._value * 1000.0  * 48.0 }))

  |> yield(name: "sum")""",
        "56new",
        tback,
        t1,
    )
    out = {}
    wh_series = []
    t_series = []

    age_series = []
    date_series = []
    for row in list(res) + list(res2):
        temp = do_query(
            """
  |> filter(fn: (r) => r["_measurement"] == "sensor.openweathermap_condition" or r["_measurement"] == "weather.56_new_road")
  |> filter(fn: (r) => r["_field"] == "temperature_valid" or r["_field"] == "temperature")
  |> filter(fn: (r) => r["domain"] == "weather")
  |> filter(fn: (r) => r["entity_id"] == "56_new_road")
  |> aggregateWindow(every: 100h, fn: mean, createEmpty: false)
  //|> map(fn: (r) => ({ r with _value: math.mMax(x:0.0, y:18.0 - r._value ) }))
  |> yield(name: "mean")""",
            "home-assistant",
            row["_time"] - datetime.timedelta(hours=24),
            row["_time"],
            extra='import "math"',
        )

        if row["_value"] > 0:
            age_series.append((tnow - row["_time"]).days)
            t_series.append(outt["_value"])
            wh_series.append(row["_value"])
        out[row["_time"].strftime("%Y-%m-%d")] = row["_value"]

    plt.figure(figsize=(12, 8))
    plt.scatter(wh_series, t_series, c=age_series, cmap="coolwarm_r")
    plt.xlabel("Wh gas used")
    plt.ylabel("mean outside temperature")
    return out


gas_use = json_cache("gas.json", get_daily_gas_use)


latitude = site["location"]["latitude"]
longitude = site["location"]["longitude"]
model_scale = 1
actual_scale = 8860 / 7632
battery_cost = 0  # take into account batter wear and tear if non-zero, in pounds
battery_lifetime_wh = 37.8e6 * 2
battery_size = 28000
maximum_charge_rate_watts = 10000
reserve_threshold = 0.25
battery_efficiency = 0.9
discharge_threshold = 0.48
discharge_price_floor = 0.27
discharge_rate_w = 10000
extra_verbose = False
battery_cost_per_wh = battery_cost / battery_lifetime_wh
print(f"battery cost per kwh=£{battery_cost_per_wh*1e3}")


def get_agile(t, outgoing=False):
    y = t.year - 2023
    if y > 0 :
        markup = site['tariffs']['agile']['scale']**y
        if t.day == 29 and t.month == 2:
            t = t.replace(day=28)
        t = t.replace(year=2023)

    v = agile_incoming_cache.get((t, outgoing))
    if v is not None:
        return v*markup
    print("miss", t, outgoing)


usage_model = {t: u for (t, u) in mean_time_of_day}


def simulate_tariff(
    name="unknown",
    actual=False,
    grid_charge=False,
    grid_discharge=False,
    battery=True,
    solar=True,
    standing=0.4201,
    agile_charge=False,
    saving_sessions_discharge=False,
    winter_agile_import=False,
    gas_hot_water=False,
    electricity_costs=None,
    color="grey",
    verbose=False,
    start = datetime.datetime.strptime(
        "2024-01-01 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z"
    ),
    end = datetime.datetime.strptime(
        "2025-01-01 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z"
    )
):
    if electricity_costs is None:
        electricity_costs = [
            {
                "start": 0,
                "end": 24,
                "import": 0.2793,
                "export": 0.15,
            },
        ]
    if verbose:
        print("run simulation", name)
    kwh_days = []
    soc = battery_size if battery else 0

    cost = 0
    cost_series = []
    solar_prod_total = 0
    days = []
    day_cost_map = {}
    halfhours = []
    soc_series = []
    solar_output_w_count = 0
    hh_count = 0
    soc_daily_lows = []
    day_costs = []
    battery_commision_date = datetime.datetime.strptime(site['powerwall']['commission_date'], '%Y-%m-%d %H:%M:%S %z')
    t = start
    for day in range((end - start).days):
        tday = t + datetime.timedelta(days=day)
        tmodel = map_to_model_date(tday)
        if 'discharge' in name and day > 400: verbose=True
        if verbose:
            print("day", day, tday, 'equivalent day in modelling period',tmodel)
        battery_today = battery and tday >= battery_commision_date

        electricity_costs_today, gas_kwh_cost, gas_sc_cost = work_out_costs_today(actual, electricity_costs,
                                                                                  tday, verbose)

        gas_hot_water_saving_active = gas_hot_water and tday.month >= 3
        kwh = 0
        day_cost = 0
        min_soc = soc
        soc_daily = []
        solar_today = 0
        slots = [tday + datetime.timedelta(minutes=x * 30) for x in range(48)]
        charge_slots, highest_outgoing = work_out_agile_charge_slots(slots, verbose) if agile_charge else [], 0
        electricity_import_cost = 0
        gas_import_cost = 0
        electricity_export_cost = 0
        gas_hot_water_saving = 2200 if gas_hot_water_saving_active else 0
        for hh in range(48):
            hh_count += 1
            time_of_day = tday + datetime.timedelta(minutes=hh * 30)
            pos, solar_prod_kwh_hh = work_out_solar_production(solar, solar_output_w_count, time_of_day, verbose)
            solar_prod_total += solar_prod_kwh_hh
            solar_today += solar_prod_kwh_hh
            kwh += solar_prod_kwh_hh
            usage_hh = usage_actual.get(time_of_day)
            usage_real = True
            if verbose and usage_hh:
                print(f"{time_of_day} actual electrical usage {usage_hh}Wh")
            if usage_hh is None:
                usage_hh = usage_model[time_of_day.strftime("%H:%M")]
                usage_real = False
                if verbose:
                    print(f"{time_of_day} modelled electrical usage {usage_hh}Wh")
            if time_of_day.hour >= 2 and gas_hot_water_saving > 0:
                discount = min(gas_hot_water_saving, usage_hh - 100)
                usage_hh -= discount
                gas_hot_water_saving -= discount
                if verbose:
                    print(
                        f"gas hot water reduced usage at {time_of_day} by  {discount}Wh to ${usage_hh}"
                    )
                assert usage_hh > 0
            net_use = usage_hh - solar_prod_kwh_hh  # net wh energy requirement for the period
            if verbose:
                print(f"{time_of_day} net electrical requirement {net_use}Wh")
            price_matches = [
                price
                for price in electricity_costs_today
                if time_of_day.hour >= price["start"] and time_of_day.hour < price["end"]
            ]
            assert len(price_matches) == 1, (price_matches, name, time_of_day, electricity_costs_today)
            price = price_matches[0]
            if verbose:
                print(f"{time_of_day} price structure {price}")
            import_cost = price["import"]
            winter_override = winter_agile_import and time_of_day.month in [11, 12, 1, 2, 3]
            if import_cost == "agile" or winter_override:
                import_cost = get_agile(time_of_day) / 100
            if price["export"] == "agile":
                export_payment = (
                    get_agile(
                        time_of_day,
                        outgoing=True,
                    )
                    / 100
                )
                if verbose:
                    print("export", time_of_day, "is", export_payment)
            else:
                export_payment = price["export"]
                if winter_override:
                    export_payment = 0.15
            if hh == 0:
                gas_kwh_day = gas_use.get(tmodel.strftime("%Y-%m-%d"), 0) / 1000 + (
                    10 if gas_hot_water_saving_active else 0
                )
                gas_cost = ( gas_sc_cost + gas_kwh_day * gas_kwh_cost)
                gas_import_cost += gas_cost
                if verbose:
                    print(f"{time_of_day} gas use {gas_kwh_day}kWh, cost £${gas_cost:.2f}")
            else:
                gas_cost = 0
            hh_cost = (standing + gas_cost) if hh == 0 else 0
            wh_from_grid = 0
            soc_delta = 0
            if verbose:
                print(f"{time_of_day} net use net use {net_use}Wh (usage={usage_hh}Wh solar={solar_prod_kwh_hh}Wh)")
            if net_use > 0:
                # we do need energy
                bat_reserve_limit = battery_size * reserve_threshold
                wh_from_battery = (
                    min(net_use, soc - bat_reserve_limit, maximum_charge_rate_watts / 2)
                    if battery_today
                    else 0
                )
                wh_from_grid += net_use - wh_from_battery
                soc_delta -= wh_from_battery
                if verbose:
                    print(f"{time_of_day} taking {wh_from_grid}Wh from grid and {wh_from_battery}Wh from battery (battery today={battery_today}) daily import cost now={electricity_import_cost}")
            else:
                # we have spare energy
                # we have spare energy
                if solar_prod_kwh_hh > 0:
                    charge_delta = (
                        min(-net_use, (battery_size - soc) / battery_efficiency)
                        if battery_today
                        else 0
                    )
                    soc_delta_charge = min(
                        maximum_charge_rate_watts / 2 - soc_delta,
                        charge_delta * battery_efficiency,
                    )
                    if verbose:
                        print(
                            "charge",
                            charge_delta,
                            "soc delta",
                            soc_delta_charge,
                            "to deal with",
                            net_use,
                        )
                    soc_delta += soc_delta_charge
                    assert soc + soc_delta <= battery_size
                    export_kwh = (-net_use - charge_delta) / 1000
                else:
                    export_kwh = 0
                export_kwh = max(export_kwh, 0)
                wh_from_grid = -export_kwh * 1000
                assert export_kwh >= 0
                hh_cost -= export_kwh * export_payment
                electricity_export_cost -= export_kwh * export_payment
            if agile_charge:
                grid_charge_now = time_of_day in [x[1] for x in charge_slots] and (
                    soc > battery_size * 0.5 and import_cost > 0.1
                )
            else:
                grid_charge_now = (
                    time_of_day.hour >= 2 and time_of_day.hour < 5 and grid_charge and battery
                )

            if grid_charge_now and battery_today:
                wh_from_grid += max(0, (
                    min(
                        (battery_size - soc) / battery_efficiency,
                        maximum_charge_rate_watts / 2,
                    )
                    - soc_delta
                ))
                add = wh_from_grid * battery_efficiency
                if verbose:
                    print(
                        "grid charge",
                        wh_from_grid,
                        "add",
                        add,
                        "soc delta was",
                        soc_delta,
                        "now",
                        soc_delta + add,
                    )
                soc_delta += add
            if wh_from_grid > 0:
                cost_electricity = import_cost * wh_from_grid / 1000
                hh_cost += cost_electricity
                electricity_import_cost += cost_electricity

            export_payment_bonus = 0
            if grid_discharge or saving_sessions_discharge:
                go = False
                if saving_sessions_discharge and (time_of_day.month == 12 and time_of_day.year == 2023) or (time_of_day.month in [1, 2] and time_of_day.year==2024):
                    go = (
                        time_of_day.month in [12, 1, 2]
                        and time_of_day.day in [7, 14, 21]
                        and (time_of_day.hour == 17 or (time_of_day.hour == 18 and time_of_day.minute < 30))
                    )
                    export_payment_bonus = 4.2
                elif grid_discharge and price["export"] == "agile":
                    go = (
                        export_payment >= highest_outgoing[6] / 100
                        and export_payment > 0.25
                    )
                elif grid_discharge:
                    go = export_payment >= discharge_price_floor
                if go:
                    if verbose:
                        print("grid discharge available at", t, "soc", soc)
                    if soc >= discharge_threshold * battery_size:
                        limit = battery_size * discharge_threshold
                        dump_amount = max(
                            0, min((soc + soc_delta) - limit, discharge_rate_w / 2)
                        )
                        if verbose:
                            print("dumping", dump_amount)
                        soc_delta -= dump_amount
                        payment = (dump_amount / 1000) * (
                            export_payment + export_payment_bonus
                        )
                        hh_cost -= payment
                        if verbose:
                            print(
                                "t",
                                t,
                                "grid dump at SOC",
                                soc,
                                "of",
                                dump_amount,
                                "payment",
                                payment,
                            )
            # if import_cost < 0:
            #     hh_cost += import_cost * 4 # run some heaters
            battery_wear_cost = battery_cost_per_wh * -min(0, soc_delta)
            hh_cost += battery_wear_cost
            cost += hh_cost
            day_cost += hh_cost

            soc_daily.append(soc)
            soc_series.append(soc)
            halfhours.append(time_of_day)

            new_soc = min(battery_size, soc + soc_delta)
            if verbose:
                print(
                    f'{name} {time_of_day} sun {pos}, solar prod {solar_prod_kwh_hh/1000:.3f}kWh{"*" if usage_real else "?"}, '
                    + f'usage {usage_hh/1000:.03f}{"*" if usage_real else "?"} net_use_house {net_use/1000:.03f} '
                    + f"grid flow {wh_from_grid/1000:.3f}kWh in@£{import_cost} ex@£{export_payment} "
                    + f"battery flow {soc_delta/1000:.3f}kWh -> {new_soc/1000:.3f}kWh min_bat "
                    + f"{min_soc/1000:.3f}kWh battery_wear=£{battery_wear_cost:0.02f} hh=£{hh_cost:0.02f} total £{cost:0.02f}"
                )

            assert new_soc < battery_size * 1.01, (new_soc, battery_size)

            soc = min(battery_size, soc)
            min_soc = min(soc, min_soc)
            soc = min(battery_size, new_soc)
            assert soc <= battery_size
            assert soc >= 0
        soc_daily_lows.append(100.0 * min(soc_daily) / battery_size)
        days.append(tday)
        kwh_days.append(kwh)
        cost_series.append(cost)
        day_costs.append(day_cost)

        day_cost_map[tday.date()] = {'electricity_import_cost':electricity_import_cost, 'gas_import_cost':gas_import_cost, "electricity_export_cost":electricity_export_cost,
                                     "solar_production": solar_today/1000}

    if actual:
        for bill in site['bills']:
            start = datetime.datetime.strptime(bill['start'], '%Y-%m-%d').date()
            end = datetime.datetime.strptime(bill['end'], '%Y-%m-%d').date()
            total = 0
            day_cost = {}
            if 'total' not in bill:
                bill['total'] = bill['electricity_import_cost'] + bill['gas_import_cost'] + bill['electricity_export_cost']
            for i in range((end-start).days):
                day = start+datetime.timedelta(days=i)
                print(day, day_cost_map[day])
                for field, x in day_cost_map.get(day, {}).items():
                    day_cost.setdefault(field, 0)
                    day_cost[field] += x
                    total += day_cost[field]
            print(f'Bill from {start} to {end}')
            for field in day_cost:
                print('\tfor '+field)
                print('\t\texpected', bill.get(field), 'actual', day_cost[field])
    prev = 0
    month_cost = [0] * 12
    months = []
    for cost, day in zip(day_costs, days):
        if day.day == 14:
            months.append(day)
        month_cost[day.month - 1] += cost
    # month_cost = month_cost[: len(months)]
    # assert len(months) == len(month_cost), (
    #     len(months),
    #     len(month_cost),
    #     months,
    #     month_cost,
    # )

    return {
        "month_cost": month_cost,
        "day_cost_map":day_cost_map,
        "annual cost": cost_series[-1],
        "name": name,
        "soc_daily_lows": soc_daily_lows,
        "color": color,
        "kwh_days": kwh_days,
        "cost_series": cost_series,
        "days": days,
        "months": months,
    }


def work_out_solar_production(solar, solar_output_w_count, time_of_day, verbose):
    pos = get_solar_position_index(time_of_day)
    if not solar:
        solar_prod_kwh_hh = 0
        solar_real = False
    else:
        if time_of_day in solar_output_w:
            solar_prod_kwh_hh = solar_output_w[time_of_day] * actual_scale
            solar_output_w_count += 1
            solar_real = True
        else:
            solar_real = False
            solar_prod_kwh_hh = 0 if pos[0] <= 0 else solar_model_table.get(pos, 0)
            # print('estimated solar production', solar_prod_kwh_hh, 'at',t1, pos)
    if verbose:
        print(time_of_day, "solar position", pos, "solar production", solar_prod_kwh_hh)
    return pos, solar_prod_kwh_hh


def work_out_agile_charge_slots(slots, verbose):
    agile_series = [(get_agile(x), x) for x in slots]
    lowest = sorted(agile_series)
    agile_outgoing_series = [get_agile(x, True) for x in slots]
    highest_outgoing = list(reversed(sorted(agile_outgoing_series)))
    charge_slots = [x for x in lowest if x[0] < 0]
    if len(charge_slots) < 5:
        charge_slots = lowest[:5]
    if verbose:
        print(
            "charge slots", [(x[0], x[1].strftime("%H:%M")) for x in charge_slots]
        )
    return charge_slots, highest_outgoing


def map_to_model_date(tday):
    tday2 = tday
    if tday2.day == 29 and tday2.month == 2:
        tday2 = tday2.replace(day=28)
    tmodel = tday2.replace(year=2023)
    return tmodel


def work_out_costs_today(actual, electricity_costs, tday, verbose=False):
    gas_sc_cost = gas_tariffs_cache.get((False, tday.strftime('%Y-%m-%d')))
    gas_kwh_cost = gas_tariffs_cache.get((True, tday.strftime('%Y-%m-%d')))
    found = None
    electricity_costs_today = electricity_costs
    for tariff in site['tariff_history']:
        tstart = datetime.datetime.strptime(tariff['start'], '%Y-%m-%d')
        tend = datetime.datetime.strptime(tariff['end'], '%Y-%m-%d')
        gname = 'gas latest'
        if tday.date() >= tstart.date() and tday.date() <= tend.date():
            tname = tariff.get('electricity_tariff')
            gname = tariff.get('gas_tariff')
            if actual and tname:
                electricity_costs_today = []
                for period in site['tariffs'][tname]['kwh_costs']:
                    electricity_costs_today.append({
                        "start": period['start'],
                        "end": period['end'],
                        "import": period['import'],
                        "export": period['export']
                    })
                found = tname
        # reconcile site.json values with database
        gas_kwh_cost_declared = site['tariffs'][gname]['kwh_costs']['import']
        gas_sc_cost_declared = site['tariffs'][gname]['standing']
        if 'ovo' not in gname:
            if gas_kwh_cost is not None and gas_kwh_cost != gas_kwh_cost_declared:
                print(f'WARNING: declared gas kwh cost={gas_kwh_cost_declared} != database value {gas_kwh_cost} on {tday}' )
            if gas_kwh_cost is None:
                gas_kwh_cost = gas_kwh_cost_declared
            if gas_sc_cost is not None and gas_sc_cost != gas_sc_cost_declared:
                print(f'WARNING: declared gas standing charge cost={gas_sc_cost_declared} != database value {gas_sc_cost}' )
            if gas_sc_cost is None:
                gas_sc_cost = gas_sc_cost_declared


    if verbose:
        print(f'{tday}: electricity tariff={electricity_costs_today} gas sc={gas_sc_cost} kwh={gas_kwh_cost}')
        if found and verbose:
            print('prevailing electricity tariff', found, 'to', tday)
    return electricity_costs_today, gas_kwh_cost, gas_sc_cost


def plot_days(results):
    m = results['day_cost_map']
    days = sorted(m.keys())
    plt.figure(figsize=(12, 8))
    for k in ['electricity_import_cost', 'gas_import_cost',
              'electricity_export_cost',
              #'solar_production'
              ]:
        plt.plot(days,[m[d][k] for d in days], label=k)

    plt.title(results['name'])
    plt.legend()
    plt.show()
    return plt

actual_results = simulate_tariff(name='actual', actual=True, verbose=False,
                                  start=t0, end=t1, saving_sessions_discharge=True, solar=True)
plot_days(actual_results)

old_results = simulate_tariff(
     name="flexible no solar no batteries", gas_hot_water=True, verbose=False, battery=False, solar=False
)
plot_days(old_results)

discharge_results = simulate_tariff(
    name="discharge flux",
    electricity_costs=site["tariffs"]["flux"]["kwh_costs"],
    grid_charge=True,
    grid_discharge=True,
    battery=True,
    solar=True,
    color="yellow",
    verbose=False,
    saving_sessions_discharge=True,
    start=t0,
)
plot_days(discharge_results)
current_results = simulate_tariff(
    name="flux",
    electricity_costs=site["tariffs"]["flux"]["kwh_costs"],
    grid_charge=True,
    battery=True,
    solar=True,
    color="green",
    verbose=False,
    saving_sessions_discharge=True,
)
plot_days(current_results)
winter_agile_result = simulate_tariff(
    name="winter agile",
    electricity_costs=site["tariffs"]["flux"]["kwh_costs"],
    winter_agile_import=True,
    grid_charge=True,
    agile_charge=True,
    battery=True,
    solar=True,
    color="blue",
    verbose=False,
    saving_sessions_discharge=True,
)

agile_results = simulate_tariff(
    name="agile",
    electricity_costs=[
        {
            "start": 0,
            "end": 24,
            "import": "agile",
            "export": "agile",
        },
    ],
    winter_agile_import=True,
    grid_charge=True,
    agile_charge=True,
    battery=True,
    solar=True,
    color="blue",
    verbose=False,
    saving_sessions_discharge=True,
)


def plot(results_list):
    f, (ax1, ax2, ax3, ax4) = matplotlib.pyplot.subplots(4, 1, sharex=True)
    f.set_figwidth(18)
    f.set_figheight(18)

    for r in results_list:
        ax1.plot(r["days"], r["cost_series"], color=r["color"], label=r["name"])
        ax1.set_ylabel("cumulative cost, £")
        ax2.plot(r["days"], r["soc_daily_lows"], color=r["color"], label=r["name"])
        ax2.set_ylabel("battery daily low %")

        if [x for x in r["kwh_days"] if x > 0]:
            ax3.plot(r["days"], r["kwh_days"])
        ax3.set_ylabel("solar production, Wh")
        #ax4.plot(r["months"], r["month_cost"], color=r["color"], label=r["name"])
        ax4.set_ylabel("monthly cost, £")
        print("plan", r["name"], r["cost_series"][-1])

    ax1.legend()
    ax2.legend()
    ax4.legend()
    return f


winter_agile_result["color"] = "pink"
f = plot(
    [
        old_results,
        current_results,
        agile_results,
        winter_agile_result,
        discharge_results,
    ]
)
f.show()
