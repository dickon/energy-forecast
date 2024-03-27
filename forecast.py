import datetime
import json
import matplotlib
import os.path
import math
import sys
import pysolar
import pickle
import pprint
from functools import cache
import traceback
import influxdb_client
import matplotlib.pyplot as plt
from influxdb_client.client.write_api import SYNCHRONOUS
from statistics import mean, median
import seaborn
import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["figure.figsize"] = (20,12)

RUN_ARCHIVE = []
TIME_STRING_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

AZIMUTH_RESOLUTION = 3
ALTITUDE_RESOLUTION = 1

SITE = json.load(open(os.path.expanduser("~/src/powerwallextract/site.json")))

def parse_time(text):
    return datetime.datetime.strptime(
    tnext, "%Y-%m-%d %H:%M:%S %z"
) 
t0 = datetime.datetime.strptime(
    "2023-01-01 00:00:00 Z", "%Y-%m-%d %H:%M:%S %z"
)  # start of modelling period
t1 = datetime.datetime.strptime(
    "2023-12-31 23:59:00 Z", "%Y-%m-%d %H:%M:%S %z"
)  # end of modelling period
tnow = datetime.datetime.strptime(
    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S Z"), "%Y-%m-%d %H:%M:%S %z"
).replace(minute=0, second=0, microsecond=0)


def parse_time(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %z")


overrides = {}
for bill in SITE["bills"]:
    bill_start = datetime.datetime.strptime(bill["start"], "%Y-%m-%d").date()
    bill_end = datetime.datetime.strptime(bill["end"], "%Y-%m-%d").date()
    num_days = (bill_end - bill_start).days
    print(bill_start, bill_end, num_days, bill)
    if bill.get("override_daily"):
        day = bill_start
        while day <= bill_end:
            day += datetime.timedelta(days=1)
            overrides[(day.month, day.day)] = {
                "export_kwh": bill["electricity_export_kwh"] / num_days,
                "import_kwh": bill["electricity_import_kwh"] / num_days,
            }

EPOCH = parse_time("1975-01-01 00:00:00 Z")


def time_string(dt):
    return dt.strftime(TIME_STRING_FORMAT)


@cache
def open_influx():
    client = influxdb_client.InfluxDBClient(
        url=(SITE["influx"]["server"]),
        token=SITE["influx"]["token"],
        org=SITE["influx"]["org"],
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    query_api = client.query_api()
    return (write_api, query_api)


@cache
def open_influx_query():
    return open_influx()[1]


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
    resl = open_influx_query().query(org=SITE["influx"]["org"], query=fquery)
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


def memoize(filename, generate, max_age_days=0):
    if os.path.exists(filename):
        mtime = os.stat(filename).st_mtime
        mtime_date = datetime.datetime.fromtimestamp(mtime)
        age = datetime.datetime.now() - mtime_date
        print("cache file", filename, "age", age)
        if age.days < max_age_days:
            try:
                with open(filename, "rb") as fp:
                    return pickle.load(fp)
            except:
                print(f"unable read {filename}; generating")
                traceback.print_exc()
    print("generating", filename)
    data = generate()
    with open(filename, "wb") as fp:
        pickle.dump(data, fp)
    return data


def generate_mean_time_of_day():
    usage_actual = {}
    time_of_day = {}
    print("querying vue data")
    for res in do_query(
        """|> filter(fn: (r) => r["_measurement"] == "energy_usage")
    |> filter(fn: (r) => r["_field"] == "usage")
    |> filter(fn: (r) => r["account_name"] == "Primary Residence")
    |> filter(fn: (r) => r["detailed"] == "False")
    |> filter(fn: (r) => r["device_name"] == "Home-TotalUsage")
    |> filter(fn: (r) => r._value > 0.0)
    |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
    """,
        "vue",
        EPOCH,
        tnow,
    ):
        usage = res["_value"] / 2
        t = res["_time"]
        record_usage(t, time_of_day, usage, usage_actual)
    print("querying powerwall data")
    prev = None
    prevt = None
    for res in do_query(
        """
            |> filter(fn: (r) => r["_measurement"] == "energy")
            |> filter(fn: (r) => r["_field"] == "energy_imported")
            |> filter(fn: (r) => r["meter"] == "load")
            |> window(every: 30m)
            |> last()
            |> duplicate(column: "_stop", as: "_time")
            |> window(every: inf)
             """,
        "powerwall",
        EPOCH,
        tnow,
    ):

        t = res["_time"]
        value = res["_value"]

        if prevt:
            deltat = t - prevt
            if deltat.seconds == 1800:
                usage = value - prev
                record_usage(t, time_of_day, usage, usage_actual)
        prevt = t
        prev = value

    return {
        "mean_time_of_day": [(t, mean(x)) for (t, x) in time_of_day.items()],
        "usage_actual": usage_actual,
    }


def record_usage(t, time_of_day, usage, usage_actual):
    usage_actual[t.isoformat()] = usage
    tday = f"{t.hour:02d}:{t.minute:02d}"
    time_of_day.setdefault(tday, list()).append(usage)


def plot_usage_scatter():
    plt.figure(figsize=(12, 8))
    plt.scatter(
        [t for t, _ in values],
        [t.hour + t.minute / 60 for t, _ in values],
        s=[v / 100 for _, v in values],
        c=[v / 100 for _, v in values],
    )


def get_solar_position_index(t):
    altitude = pysolar.solar.get_altitude(
        SITE["location"]["latitude"],
        SITE["location"]["longitude"],
        t + datetime.timedelta(minutes=15),
    )
    azimuth = pysolar.solar.get_azimuth(
        SITE["location"]["latitude"],
        SITE["location"]["longitude"],
        t + datetime.timedelta(minutes=15),
    )
    return ALTITUDE_RESOLUTION * round(
        altitude / ALTITUDE_RESOLUTION
    ), AZIMUTH_RESOLUTION * round(azimuth / AZIMUTH_RESOLUTION)


def title(message):
    print()
    print()
    print(message)
    print('='*len(message))
    print

def generate_solar_model(verbose=True):
    title('generating solar model')
    bucket = "56new"

    title("looking for clear skies")
    clear_skies = set()
    cloud_series = []
    cloud_t = []
    for entry in do_query(
            """
    |> filter(fn: (r) => r["entity_id"] == "openweathermap_cloud_coverage")
    |> filter(fn: (r) => r["_field"] == "value")
    |> filter(fn: (r) => r["_measurement"] == "%")
    |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
     """,
            "home-assistant",
            EPOCH,
            tnow,
        ):
        pos = get_solar_position_index(entry['_time'])
        if verbose:
            print('cloud',entry, 'sun pos', pos)
        if pos[0] > 0:
            cloud_series.append(entry['_value'])
            cloud_t.append(entry['_time'])
            if entry['_value'] <= 20:
                clear_skies.add(entry['_time'])
    clouds = pd.DataFrame(cloud_series, index=cloud_t)
    title('clear skies')
    print(clouds)
    ax = clouds.plot()
    ax.get_figure().savefig('clouds.png')
    ax2 = seaborn.displot(clouds, kind='ecdf')
    ax2.figure.savefig('clouds_dist.png')
    pprint.pprint(clear_skies)
    solar_output_w = {}
    solar_pos_model = {}
    solar_pos_table = {}
    base = None
    t = tcommission = parse_time(SITE["solar"]["commission_date"])
    exclusions = [
        (parse_time(rec["start"]), parse_time(rec["end"]))
        for rec in SITE["solar"]["data_exclusions"]
    ]
    overridden = {}

    def populate(t, usage, actual=True):
        if t in overridden and not actual:
            print('overriden', t, 'usage', usage, 'by', overridden[t])
            return
        pos = get_solar_position_index(t)
        if t in clear_skies:
            solar_pos_table.setdefault(pos, list())
            solar_pos_table[pos].append(max(0, usage))
            print('populate', repr(t), usage, 'actual=',actual)
        if actual:
            solar_output_w[t] = max(0, usage)

    used_kwh_day = sum([x[1] for x in mean_time_of_day]) / 1000
    
    title('doing estimations from bills')
    print("daily kwh used", used_kwh_day)
    for day, dayover in overrides.items():
        generated_kwh_day = dayover["export_kwh"] + used_kwh_day - dayover["import_kwh"]
        print("day", day, "generation est", generated_kwh_day, "based on", dayover)
        weightings = []
        weightings_map = {}
        for hh in range(48):
            t = datetime.datetime(
                year=2023, month=day[0], day=day[1], tzinfo=datetime.timezone.utc
            ) + datetime.timedelta(minutes=30 * hh)
            pos = get_solar_position_index(t)
            if pos[0] > 5:
                weighting = pos[0] + (30 if hh > 26 and hh < 32 else 0)
            else:
                weighting = 0
            weightings.append(weighting)
            weightings_map[t] = weighting
        tot_weight = sum(weightings)
        for t in weightings_map.keys():
            est_gen_hh_wh = 1000 * weightings_map[t] * generated_kwh_day / tot_weight
            populate(t, est_gen_hh_wh, actual=False)
            overridden[t] = ('day override', est_gen_hh_wh)
    if 1:
        prev = None
        prevt = None
        title("querying powerwall solar data")
        for res in do_query(
            """
            |> filter(fn: (r) => r["_measurement"] == "energy")
                |> filter(fn: (r) => r["_field"] == "energy_exported")
                |> filter(fn: (r) => r["meter"] == "solar")
                    |> window(every: 30m)
                    |> last()
                    |> duplicate(column: "_stop", as: "_time")
                    |> window(every: inf) """,
            "powerwall",
            EPOCH,
            tnow,
        ):
            t = res["_time"]
            value = res["_value"]

            if prevt:
                deltat = t - prevt
                if deltat.seconds == 1800:
                    usage = value - prev
                    print('powerwall solar', t, 'usage', usage)
                    populate(t, usage)
                else:
                    print('powerwall solar', t, 'gap', deltat, 'ignored usage', usage)
                    
            prevt = t
            prev = value
    title("querying vue data")
    for res in do_query(
        """
            |> filter(fn: (r) => r["_field"] == "usage")
            |> filter(fn: (r) => r["account_name"] == "Primary Residence")
            |> filter(fn: (r) => r["detailed"] == "True")
            |> filter(fn: (r) => r["device_name"] == "SolarEdge Inverter")
            |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
             """,
        "vue",
        EPOCH,
        parse_time(SITE['powerwall']['commission_date']),
        verbose=False,
    ):
        print('vue solar', res['_time'], res['_value'])
        populate(res["_time"], -res["_value"] / 2)


    title('filling model')
    
    solar_pos_model = { k:max(v) for k,v in solar_pos_table.items()}
    originals =solar_pos_model.keys()
    for azi in range(0, 360, AZIMUTH_RESOLUTION):
        print("smoothing azimuth", azi)
        for alt in range(0, 70, ALTITUDE_RESOLUTION):
            res = []
            pos = (alt, azi)
            if pos in originals:
                continue
            print('filling',pos)
            mindist = None
            minv = None
            minpos = None
            for altpos, altvs in solar_pos_table.items():
                altv = max(altvs)
                dist = math.sqrt(
                    (altpos[0] - pos[0]) ** 2  + (altpos[1] - pos[1]) ** 2
                )
                if (mindist is None or dist < mindist):
                    if verbose:
                        print('pos',pos,'from',altpos,'dist',dist)
                    minv = altv
                    mindist = dist
                    minpos = altpos
            if verbose:
                print('pos', pos, 'mindist',mindist,'minv',minv,'from',minpos)
            if minv:
                solar_pos_model[pos] = minv
    record = {
        "solar_pos_model": solar_pos_model,
        "solar_output_w": solar_output_w,
        "clouds": pd.Series(cloud_series, index=cloud_t)
    }
    return record


txover = datetime.datetime.strptime(
    SITE["powerwall"]["commission_date"], "%Y-%m-%d %H:%M:%S %z"
)
t_solar_export_payments_start = datetime.datetime.strptime(
    SITE["solar"]["export_payments_start_date"], "%Y-%m-%d %H:%M:%S %z"
)
data = memoize("kwh_use_time_of_day.pickle", generate_mean_time_of_day)
mean_time_of_day = data["mean_time_of_day"]

solar_model_record = memoize(
    "solar_model.pickle", generate_solar_model, max_age_days=14
)
solar_output_w = solar_model_record ['solar_output_w']
solar_pos_model = solar_model_record["solar_pos_model"]
solar_model_table = solar_pos_model.items()

if 0:
    title('solar_output_w')
    pprint.pprint(solar_output_w)


usage_actual_text = data["usage_actual"]
usage_actual = {
    datetime.datetime.fromisoformat(x[0]): x[1] for x in usage_actual_text.items()
}
values = [x for x in usage_actual.items()]

if "usage" in sys.argv:
    plot_usage_scatter()


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
        verbose=True,
    )
    for res in gas_res:
        t = res["_time"]
        v = res["_value"]
        rec.setdefault("gas", dict())

        rec["gas"][t.date().strftime("%Y-%m-%d")] = v
    gas_sc = do_query(
        f"""
      |> filter(fn: (r) => r["_measurement"] == "standing_charge")
      |> filter(fn: (r) => r["fuel"] == "gas")
      |> filter(fn: (r) => r["product"] == "VAR-22-11-01")""",
        "tariffs",
        t0,
        t1,
        verbose=True,
    )
    for res in gas_sc:
        t = res["_time"]
        v = res["_value"]
        rec.setdefault("gas standing", dict())
        rec["gas standing"][t.date().strftime("%Y-%m-%d")] = v

    return rec


tariffs = memoize("tariffs.pickle", get_tariffs)
agile_incoming_cache = {}
gas_tariffs_cache = {}
for outgoing in [True, False]:
    k = "agile " + ("outgoing" if outgoing else "incoming")
    for t, value in tariffs[k].items():
        agile_incoming_cache[(datetime.datetime.fromisoformat(t), outgoing)] = value
for t, value in tariffs["gas"].items():
    gas_tariffs_cache[(True, datetime.datetime.fromisoformat(t))] = value
for t, value in tariffs["gas standing"].items():
    gas_tariffs_cache[(False, datetime.datetime.fromisoformat(t))] = value


def plot_model(m, verbose=False):
    if verbose:
        title('solar model')
        pprint.pprint(m)
    altitude = [x[0][0] for x in m]
    azimuth = [x[0][1] for x in m]
    pow = [x[1] for x in m]
    p = plt.scatter(azimuth, altitude, [x / 60 for x in pow], pow)
    plt.ylim([0, 70])
    plt.xlim((50, 300))
    return p

def plot_solar_azimuth_altitude_chart(solar_model_table):
    plt.figure(figsize=(12, 8))
    p = plot_model(solar_model_table)
    plt.title("solar AC output power over 30 minutes")
    plt.xlabel("azimuth")
    plt.ylabel("altitude")
    plt.colorbar()
    plt.savefig("solarmodel.png")
    return plt


plot_solar_azimuth_altitude_chart(solar_model_table).show()



def plot_solar_times():
    values = [x for x in solar_output_w.items() if x[1] > 20]
    plt.figure(figsize=(12, 8))
    plt.scatter(
        [t for t, _ in values],
        [t.hour + t.minute / 60 for t, _ in values],
        s=[v / 100 for _, v in values],
        c=[v * 2/1000 for _, v in values],
    )
    plt.xlabel("date")
    plt.ylabel("time of day")
    plt.colorbar(label='kW')
    plt.savefig('solartimes.png')
    print('generated solartimes.png')

plot_solar_times()

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

        age_series.append((tnow - row["_time"]).days)
        t_series.append(max(0, row["_value"]))
        wh_series.append(max(0, row["_value"]))
        out[row["_time"].strftime("%Y-%m-%d")] = row["_value"]

    plt.figure(figsize=(12, 8))
    plt.scatter(wh_series, t_series, c=age_series, cmap="coolwarm_r")
    plt.xlabel("Wh gas used")
    plt.ylabel("mean outside temperature")
    return out


gas_use = memoize("gas.pickle", get_daily_gas_use)


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
battery_cost_per_wh = battery_cost / battery_lifetime_wh
print(f"battery cost per kwh=£{battery_cost_per_wh*1e3}")


def get_agile(t, outgoing=False):
    y = t.year - 2023
    if y > 0:
        markup = SITE["tariffs"]["agile"]["scale"] ** y
        if t.day == 29 and t.month == 2:
            t = t.replace(day=28)
        t = t.replace(year=2023)

    v = agile_incoming_cache.get((t, outgoing))
    if v is not None:
        return v * markup
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
    start=tnow,
    end=tnow + datetime.timedelta(days=365),
    balance=0.0,
):
    title('modelling '+name)
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
    cost = balance
    cost_series = []
    solar_prod_total = 0
    days = []
    day_cost_map = {}
    halfhours = []
    soc_series = []
    hh_count = 0
    soc_daily_lows = []
    day_costs = []
    battery_commision_date = datetime.datetime.strptime(
        SITE["powerwall"]["commission_date"], "%Y-%m-%d %H:%M:%S %z"
    )
    if actual:
        sessions = do_query("""
    |> filter(fn: (r) => r["_measurement"] == "savings session")
    """, "56new", start, end)
    t = start
    for day in range((end - start).days):
        tday = t + datetime.timedelta(days=day)
        tmodel = map_to_model_date(tday)
        # if 'discharge' in name and day > 400: verbose=True
        if verbose:
            print("day", day, tday, "equivalent day in modelling period", tmodel)
        battery_today = battery and tday >= battery_commision_date

        electricity_costs_today, gas_kwh_cost, gas_sc_cost = work_out_prices_today(
            actual, electricity_costs, tday, verbose
        )

        gas_hot_water_saving_active = gas_hot_water and tday.month >= 3
        kwh = 0
        day_cost = 0
        min_soc = soc
        soc_daily = []
        solar_today = 0
        slots = [tday + datetime.timedelta(minutes=x * 30) for x in range(48)]
        charge_slots, highest_outgoing = (
            work_out_agile_charge_slots(slots, verbose) if agile_charge else []
        ), 0
        if verbose and agile_charge:
            print("agile charge at", charge_slots, "highest outgoing", highest_outgoing)
        electricity_import_cost = 0
        gas_import_cost = 0
        electricity_export_cost = 0
        gas_hot_water_saving = 2200 if gas_hot_water_saving_active else 0
        for hh in range(48):
            hh_count += 1
            time_of_day = tday + datetime.timedelta(minutes=hh * 30)
            pos, solar_prod_wh_hh, _ = work_out_solar_production(
                time_of_day, verbose=verbose, solar=solar
            )
            solar_prod_total += solar_prod_wh_hh
            solar_today += solar_prod_wh_hh
            kwh += solar_prod_wh_hh
            usage_hh, usage_real = work_out_electricity_usage(time_of_day, verbose)
            if time_of_day.hour >= 2 and gas_hot_water_saving > 0:
                discount = min(gas_hot_water_saving, usage_hh - 100)
                usage_hh -= discount
                gas_hot_water_saving -= discount
                if verbose:
                    print(
                        f"gas hot water reduced usage at {time_of_day} by  {discount}Wh to ${usage_hh}"
                    )
                assert usage_hh > 0
            net_use = (
                usage_hh - solar_prod_wh_hh
            )  # net wh energy requirement for the period
            if verbose:
                print(f"{time_of_day} net electrical requirement {net_use}Wh")
            price = work_out_prices_now(
                electricity_costs_today,
                name,
                time_of_day,
                verbose,
                winter_agile_import,
                saving_sessions_discharge and not actual,
            )
            if hh == 0:
                gas_kwh_day = gas_use.get(tmodel.strftime("%Y-%m-%d"), 0) / 1000 + (
                    10 if gas_hot_water_saving_active else 0
                )
                gas_cost = gas_sc_cost + gas_kwh_day * gas_kwh_cost
                gas_import_cost += gas_cost
                if verbose:
                    print(
                        f"{time_of_day} gas use {gas_kwh_day}kWh, cost £${gas_cost:.2f}"
                    )
            else:
                gas_cost = 0
            if verbose:
                print(
                    f"{time_of_day} net use net use {net_use}Wh (usage={usage_hh}Wh solar={solar_prod_wh_hh}Wh)"
                )
            soc_delta, wh_from_grid = (
                handle_energy_supply(battery_today, net_use, soc, time_of_day, verbose)
                if net_use > 0
                else handle_energy_excess(
                    battery_today, net_use, soc, solar_prod_wh_hh, verbose
                )
            )
            soc_delta, wh_from_grid = handle_grid_charge(
                agile_charge,
                battery,
                battery_today,
                charge_slots,
                grid_charge,
                price["import"],
                soc,
                soc_delta,
                time_of_day,
                verbose,
                wh_from_grid,
            )

            soc_delta, wh_from_grid = handle_grid_discharge(
                wh_from_grid,
                price["export"],
                grid_discharge,
                highest_outgoing,
                saving_sessions_discharge,
                soc,
                soc_delta,
                time_of_day,
                price["export"] == "agile",
                verbose=verbose,
            )
            export_payment_bonus = 0
            battery_wear_cost = battery_cost_per_wh * -min(0, soc_delta)
            electricty_cost_hh = max(0, wh_from_grid) * price["import"] / 1000
            export_payment_hh = min(0, wh_from_grid) * price["export"] / 1000
            hh_cost = (
                ((standing + gas_cost) if hh == 0 else 0)
                + electricty_cost_hh
                + export_payment_hh
                + battery_wear_cost
            )
            if actual and saving_sessions_discharge:
                for rec in sessions:
                    if rec['_time'] >= time_of_day and rec['_time'] < time_of_day+datetime.timedelta(minutes=30):
                        print('saving session at', time_of_day,'is',rec)
                        hh_cost -= rec['_value']
            cost += hh_cost
            electricity_import_cost += electricty_cost_hh
            electricity_export_cost += export_payment_hh
            day_cost += hh_cost

            soc_daily.append(soc)
            soc_series.append(soc)
            halfhours.append(time_of_day)

            new_soc = min(battery_size, soc + soc_delta)
            if verbose:
                print(
                    f'{name} {time_of_day} sun {pos}, solar prod {solar_prod_wh_hh/1000:.3f}kWh{"*" if usage_real else "?"}, '
                    + f'usage {usage_hh/1000:.03f}{"*" if usage_real else "?"} net_use_house {net_use/1000:.03f} '
                    + f"grid flow {wh_from_grid/1000:.3f}kWh in@£{price['import']} ex@£{price['export']} "
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
        override_today = overrides.get((tday.month, tday.day))
        if override_today and actual:
            electricity_export_cost = -override_today["export_kwh"] * price["export"]
            electricity_import_cost = (
                override_today["import_kwh"] * price["import"] + standing
            )
        day_cost_map[tday.date()] = {
            "electricity_import_cost": electricity_import_cost,
            "gas_import_cost": gas_import_cost,
            "electricity_export_cost": electricity_export_cost,
            "solar_production": solar_today / 1000,
        }

    if actual:
        compare_with_bills(day_cost_map)
    prev = 0
    # month_cost = [[0] * 12]
    # months = []
    # for cost, day in zip(day_costs, days):
    #     if day.day == 14:
    #         months.append(day)
    #     month_cost[day.month - 1 + 12*(day.year - 2023)] += cost
    # month_cost = month_cost[: len(months)]
    # assert len(months) == len(month_cost), (
    #     len(months),
    #     len(month_cost),
    #     months,
    #     month_cost,
    # )

    return {
        "day_cost_map": day_cost_map,
        "day_costs" :  pd.DataFrame({name:day_costs}, index=days), 
        "final_cost": cost_series[-1],
        "name": name,
        "soc_daily_lows": pd.DataFrame({name:soc_daily_lows}, index=days),
        "color": color,
        "kwh_days": pd.DataFrame({name:kwh_days}, index=days),
        "cost_series": pd.DataFrame({name:cost_series}, index=days),
        "days": days,
    }


def simulate_tariff_and_store(name, **params):
    results = simulate_tariff(name, **params)
    RUN_ARCHIVE.append(results)
    plot_days(results, name)
    final_cost =  results['final_cost']
    print(name, 'final cost', final_cost)
    return final_cost

def in_savings_session(time_of_day):
    return (
        (time_of_day.month == 12 and time_of_day.year >= 2023)
        or (time_of_day.month in [1, 2] and time_of_day.year >= 2024)
    ) and (
        time_of_day.month in [12, 1, 2]
        and time_of_day.day in [7, 14, 21, 28]
        and (
            time_of_day.hour == 17
            or (time_of_day.hour == 18 and time_of_day.minute < 30)
        )
    )


def handle_grid_discharge(
    wh_from_grid,
    export_payment,
    grid_discharge,
    highest_outgoing,
    saving_sessions_discharge,
    soc,
    soc_delta,
    time_of_day,
    agile,
    verbose=False,
):
    if grid_discharge or saving_sessions_discharge:
        go = False
        if saving_sessions_discharge:
            go = in_savings_session(time_of_day)
        elif grid_discharge and agile:
            go = export_payment >= highest_outgoing[6] / 100
            if go and verbose:
                print("agile discharge at", time_of_day, "price", export_payment)

        elif grid_discharge and not saving_sessions_discharge:
            go = export_payment >= discharge_price_floor
        if go:
            if verbose:
                print("grid discharge available at", time_of_day, "soc", soc)
            if soc >= discharge_threshold * battery_size:
                limit = battery_size * discharge_threshold
                dump_amount = max(
                    0, min((soc + soc_delta) - limit, discharge_rate_w / 2)
                )
                if verbose:
                    print("dumping", dump_amount)
                soc_delta -= dump_amount
                if verbose:
                    print(
                        "t",
                        t,
                        "grid dump at SOC",
                        soc,
                        "of",
                        dump_amount,
                    )
                return soc_delta, wh_from_grid - dump_amount
    return soc_delta, wh_from_grid


def handle_grid_charge(
    agile_charge,
    battery,
    battery_today,
    charge_slots,
    grid_charge,
    import_cost,
    soc,
    soc_delta,
    time_of_day,
    verbose,
    wh_from_grid,
):
    if agile_charge:
        grid_charge_now = time_of_day in [x[1] for x in charge_slots] and (
            soc > battery_size * 0.5 and import_cost > 0.1
        )
    else:
        grid_charge_now = (
            time_of_day.hour >= 2 and time_of_day.hour < 5 and grid_charge and battery
        ) and time_of_day.month in [1, 2, 10, 11, 12]
    if grid_charge_now and battery_today:
        wh_from_grid += max(
            0,
            (
                min(
                    (battery_size - soc) / battery_efficiency,
                    maximum_charge_rate_watts / 2,
                )
                - soc_delta
            ),
        )
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
    return soc_delta, wh_from_grid


def handle_energy_excess(battery_today, net_use, soc, solar_prod_wh_hh, verbose):
    # we have spare energy
    if solar_prod_wh_hh > 0:
        charge_delta = (
            min(-net_use, (battery_size - soc) / battery_efficiency)
            if battery_today
            else 0
        )
        soc_delta_charge = min(
            maximum_charge_rate_watts / 2,
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
        assert soc + soc_delta_charge <= battery_size
        export_kwh = (-net_use - charge_delta) / 1000
    else:
        export_kwh = 0
        soc_delta_charge = 0
    export_kwh = max(export_kwh, 0)
    assert export_kwh >= 0
    wh_from_grid = -export_kwh * 1000
    return soc_delta_charge, wh_from_grid


def handle_energy_supply(battery_today, net_use, soc, time_of_day, verbose):
    """Called when we do need energy. Supply `net_use` Wh from either grid of battery, given the `time_of_day`

    Iff `battery_today` we can use the battery, which has capacity `soc` (in Wh).
    Returns the Wh change in battery level and the Wh from grid.
    """
    bat_reserve_limit = battery_size * reserve_threshold
    wh_from_battery = (
        min(net_use, soc - bat_reserve_limit, maximum_charge_rate_watts / 2)
        if battery_today
        else 0
    )
    wh_from_grid = net_use - wh_from_battery
    soc_delta = -wh_from_battery
    if verbose:
        print(
            f"{time_of_day} taking {wh_from_grid}Wh from grid and {wh_from_battery}Wh from battery (battery today={battery_today})"
        )
    return soc_delta, wh_from_grid


def compare_with_bills(day_cost_map):
    for bill in SITE["bills"]:
        start = datetime.datetime.strptime(bill["start"], "%Y-%m-%d").date()
        end = datetime.datetime.strptime(bill["end"], "%Y-%m-%d").date()
        total = 0
        day_cost = {}
        if "total" not in bill:
            bill["total"] = (
                bill["electricity_import_cost"]
                + bill["gas_import_cost"]
                + bill["electricity_export_cost"]
            )
        for i in range((end - start).days):
            day = start + datetime.timedelta(days=i)
            print(day, day_cost_map[day])
            for field, x in day_cost_map.get(day, {}).items():
                day_cost.setdefault(field, 0)
                day_cost[field] += x
                total += day_cost[field]
        print(f"Bill from {start} to {end}")
        for field in day_cost:
            print("\tfor " + field)
            print("\t\texpected", bill.get(field), "actual", day_cost[field])


def work_out_prices_now(
    electricity_costs_today,
    name,
    time_of_day,
    verbose=False,
    winter_agile_import=False,
    savings_session_discharge=False,
):
    price_matches = [
        price
        for price in electricity_costs_today
        if time_of_day.hour >= price["start"] and time_of_day.hour < price["end"]
    ]
    assert len(price_matches) == 1, (
        price_matches,
        name,
        time_of_day,
        electricity_costs_today,
    )
    price = dict(**price_matches[0])
    if verbose:
        print(f"{time_of_day} price structure {price}")
    winter_override = winter_agile_import and time_of_day.month in [11, 12, 1, 2, 3]
    if price["import"] == "agile" or winter_override:
        price["import"] = get_agile(time_of_day) / 100
    if price["export"] == "agile":
        price["export"] = (
            get_agile(
                time_of_day,
                outgoing=True,
            )
            / 100
        )
        if verbose:
            print("export", time_of_day, "is", price["export"])
    # else:
    #     if winter_override:
    #         price['export'] = export_payment
    if time_of_day < t_solar_export_payments_start:
        price["export"] = 0
    if savings_session_discharge and in_savings_session(time_of_day):
        price["export"] += 2
    return price


def work_out_electricity_usage(time_of_day, verbose):
    usage_hh = usage_actual.get(time_of_day)
    usage_real = True
    if verbose and usage_hh:
        print(f"{time_of_day} actual electrical usage {usage_hh}Wh")
    if usage_hh is None:
        usage_hh = usage_model[time_of_day.strftime("%H:%M")]
        usage_real = False
        if verbose:
            print(f"{time_of_day} modelled electrical usage {usage_hh}Wh")
    return usage_hh, usage_real


def work_out_solar_production(time_of_day, verbose=True, solar=True, use_records=True, realistic_sunshine_ratio=0.6):
    pos = get_solar_position_index(time_of_day)
    if not solar:
        solar_prod_kwh_hh = 0
        from_record = True
    else:
        if time_of_day in solar_output_w and use_records:
            solar_prod_kwh_hh = solar_output_w[time_of_day]
            from_record = True
        else:
            if verbose:
                print('solar model',pos, solar_pos_model.get(pos))
            solar_prod_kwh_hh = (
                solar_pos_model.get(pos, 0)
                * SITE["solar"].get("actual_scale_output", 1)
                / SITE["solar"].get("model_scale_output", 1)
            )*realistic_sunshine_ratio
            
            from_record = False
    if verbose:
        print(time_of_day, "solar position", pos, "solar production", solar_prod_kwh_hh, 'from record', from_record, 'use_records', use_records)
    assert solar_prod_kwh_hh is not None
    return pos, solar_prod_kwh_hh, from_record


def generate_solar_production(start="2023-01-06 00:00:00 Z", end=None, verbose=False):
    title('working out solar production and model')
    t = datetime.datetime.strptime(start, "%Y-%m-%d %H:%M:%S %z")
    end = datetime.datetime.strptime(end, "%Y-%m-%d %H:%M:%S %z") if end else tnow
    series_real = []
    real_records  = []
    series_model = []
    series_t = []
    series_t_real = []
    while t < end:
        wh_day_records = 0
        wh_day_model = 0
        records  = 0
        for hh in range(48):
            time_of_day = t + datetime.timedelta(minutes=30 * hh)
            pos, wh_from_records, from_records = work_out_solar_production(
                time_of_day, use_records=True, verbose=False, realistic_sunshine_ratio=1.0
            )
            pos2, wh_from_model, from_records_2 = work_out_solar_production(
                time_of_day, use_records=False, verbose=False, realistic_sunshine_ratio=1.0
            )
            assert from_records_2 == False
            if verbose:
                print(f'{t} {pos} wh_rec={wh_from_records} from_records={from_records} wh_model={wh_from_model}')
            if from_records:
                records += 1
            wh_day_records += wh_from_records
            wh_day_model += wh_from_model
            
        if verbose:
            print('dailysolar',t, 'model', wh_day_model, 'actual', wh_day_records, 'with records=',records, time_of_day in solar_output_w)

        series_t.append(t)
        series_model.append(wh_day_model)
        if records > 12: # only store the actual if we mostly got it from real records
            series_t_real.append(t)
            series_real.append(wh_day_records)
            real_records.append(records)
        t += datetime.timedelta(days=1)
    if verbose:
        title('real solar output:')
    tot_solar = 0
    for t, kwh in zip(series_t, series_real):
        if verbose:
            print(t, '{:4.1f}'.format(kwh), '*'*int(kwh))
        tot_solar += kwh
    title('total solar generation')
    print('total solar generator', tot_solar)

    real = pd.DataFrame({'real': series_real},  index=series_t_real)
    model = pd.DataFrame({'model':series_model}, index=series_t)
    records = pd.DataFrame({'records': real_records},  index=series_t_real)

    title('solar daily real vs model')
    combined = pd.concat([real, model, records], axis=1)
    return combined



def plot_solar_production():
    bucket_description = 'weekly'
    def bucketround(t):
        return datetime.datetime(year=t.year, month=t.month, day =((t.day-1) // 7)*7+1)

    combined = memoize('solar_production.pickle', generate_solar_production)
    
    combined['actual to max percentage'] = combined['real'] / combined['model']
    title('daily solar production analysis')
    print(combined.to_string())
    title('mohtly solar production analysis')
    combined['real_kwh'] = combined['real'] / 1000.0
    combined['model_kwh'] = combined['model'] / 1000.0

    monthlies = combined.groupby(combined.index.map(bucketround)).mean()
    print(monthlies.to_string())
    clouds = solar_model_record['clouds'].loc[solar_model_record['clouds'].index > '2023-01-01 00:00:00']
    cloud_monthlies = 1 - clouds.groupby(clouds.index.map(bucketround)).mean() / 100
    title('cloud boxes')
    print(cloud_monthlies.to_string())
    plt.figure(figsize=(12, 8))
    f, axs = matplotlib.pyplot.subplots(2, 1, sharex=True)
    combined.plot(ax=axs[0], y='real_kwh', style='.')
    #axs[0].scatter(series_t_real, series_real, label="real readings", marker='+', c='black')
    combined.plot.line(ax=axs[0], y='model_kwh')
    axs[0].set_xlabel("date")
    axs[0].set_ylabel("daily kWh output")
    axs[0].legend()

    fig = combined.plot(ax=axs[1], y='actual to max percentage', label='daily basis')
    monthlies.plot(ax=axs[1], y='actual to max percentage', label=bucket_description + ' basis')
    cloud_monthlies.plot(ax=axs[1], y='cloud percentage', label=bucket_description +' mean non-cloud cover, openweathermap', color='#909090', legend=True)
    axs[1].set_ylabel("Ratio of reality to maximum sunshine model")
    fig.get_figure().savefig('solaractual_to_max.png')

    f.savefig("dailysolar.png")

    fig = combined.plot(y='real_kwh', style='.')
    fig.set_ylabel('daily kWh solar production')
    fig.get_figure().savefig('solar_actual.png')


plot_solar_production()

def work_out_agile_charge_slots(slots, verbose):
    agile_series = [(get_agile(x), x) for x in slots]
    lowest = sorted(agile_series)
    agile_outgoing_series = [get_agile(x, True) for x in slots]
    highest_outgoing = list(reversed(sorted(agile_outgoing_series)))
    charge_slots = [x for x in lowest if x[0] < 0]
    if len(charge_slots) < 6:
        charge_slots = lowest[:6]
    if verbose:
        print("charge slots", [(x[0], x[1].strftime("%H:%M")) for x in charge_slots])
    return charge_slots, highest_outgoing


def map_to_model_date(tday):
    tday2 = tday
    if tday2.day == 29 and tday2.month == 2:
        tday2 = tday2.replace(day=28)
    tmodel = tday2.replace(year=2023)
    return tmodel


def work_out_prices_today(actual, electricity_costs, tday, verbose=False):
    gas_sc_cost = gas_tariffs_cache.get((False, tday.strftime("%Y-%m-%d")))
    gas_kwh_cost = gas_tariffs_cache.get((True, tday.strftime("%Y-%m-%d")))
    found = None
    electricity_costs_today = electricity_costs
    for tariff in SITE["tariff_history"]:
        tstart = datetime.datetime.strptime(tariff["start"], "%Y-%m-%d")
        tend = datetime.datetime.strptime(tariff["end"], "%Y-%m-%d")
        gname = "gas latest"
        if tday.date() >= tstart.date() and tday.date() <= tend.date():
            tname = tariff.get("electricity_tariff")
            gname = tariff.get("gas_tariff")
            if actual and tname:
                electricity_costs_today = []
                for period in SITE["tariffs"][tname]["kwh_costs"]:
                    electricity_costs_today.append(
                        {
                            "start": period["start"],
                            "end": period["end"],
                            "import": period["import"],
                            "export": period["export"],
                        }
                    )
                found = tname
        # reconcile site.json values with database
        gas_kwh_cost_declared = SITE["tariffs"][gname]["kwh_costs"]["import"]
        gas_sc_cost_declared = SITE["tariffs"][gname]["standing"]
        if "ovo" not in gname:
            if gas_kwh_cost is not None and gas_kwh_cost != gas_kwh_cost_declared:
                print(
                    f"WARNING: declared gas kwh cost={gas_kwh_cost_declared} != database value {gas_kwh_cost} on {tday}"
                )
            if gas_kwh_cost is None:
                gas_kwh_cost = gas_kwh_cost_declared
            if gas_sc_cost is not None and gas_sc_cost != gas_sc_cost_declared:
                print(
                    f"WARNING: declared gas standing charge cost={gas_sc_cost_declared} != database value {gas_sc_cost}"
                )
            if gas_sc_cost is None:
                gas_sc_cost = gas_sc_cost_declared

    if verbose:
        print(
            f"{tday}: electricity tariff={electricity_costs_today} gas sc={gas_sc_cost} kwh={gas_kwh_cost}"
        )
        if found and verbose:
            print("prevailing electricity tariff", found, "to", tday)
    return electricity_costs_today, gas_kwh_cost, gas_sc_cost


def plot_days(results, name):
    m = results["day_cost_map"]
    days = sorted(m.keys())
    plt.figure(figsize=(12, 8))
    for k in [
        "electricity_import_cost",
        "gas_import_cost",
        "electricity_export_cost",
        #'solar_production'
    ]:
        plt.plot(days, [m[d][k] for d in days], label=k)
    plt.title(results["name"])
    plt.legend()
    plt.show()
    plt.savefig(name + ".png")
    return plt


def plot_simulations(results_list):
        
    f, axs = matplotlib.pyplot.subplots(4, 1, sharex=True)
    f.set_figwidth(18)
    f.set_figheight(18)
    for i, field in enumerate(['day_costs', 'soc_daily_lows','kwh_days',  'cost_series']):
        df = pd.concat([r[field] for r in results_list], axis=1)
        df.plot(ax=axs[i])
        title(field)
        print(df.to_string())
        axs[i].legend()
    # for r in results_list:
    #     title('rendering '+r['name'])
    #     print('day costs', r['day_costs'])
    #     #r['day_costs'].plot(ax=ax1,color=r['color'])
    #     ax1.set_ylabel("cumulative cost, £")
    #     print('soc daily lows', r['soc_daily_lows'])
    #     r['soc_daily_lows'].plot(ax=ax2, color=r["color"])
    #     ax2.set_ylabel("battery daily low %")

    #     if [x for x in r["kwh_days"] if x > 0]:
    #         r['kwh_days'].plot(ax=ax3, label=r['name'])
    #     ax3.set_ylabel("solar production, Wh")
    #     month_series=r['day_costs'].groupby([lambda x: datetime.datetime(year=x.year, month=x.month,day=14, hour=12,minute=0,second=0)]).sum()
    #     print(month_series)
    #     month_series.plot(ax=ax4, color=r["color"])
    #     ax4.set_ylabel("monthly cost, £")
    #     print("plan", r["name"], repr(r["cost_series"].tail(1)))

    f.savefig("run.png")


def run_simulations():

    balance_base_line = simulate_tariff_and_store(
        name="baseline without batteries",
        actual=True,
        verbose=True,
        grid_charge=False,
        start=t0,
        end=tnow,
        grid_discharge=False,
        battery = False,
        saving_sessions_discharge=False,
        solar=False
    )
    balance = simulate_tariff_and_store(
        name="actual",
        actual=True,
        verbose=False,
        grid_charge=True,
        start=t0,
        end=tnow,
        grid_discharge=True,
        saving_sessions_discharge=True,
        solar=True,
    )


    title("Current balance")
    print('with batteries and solar=',balance, 'base-line without batteries and without solar=', balance_base_line)

    title("Simulations")
    simulate_tariff_and_store(
        name="discharge flux",
        balance=balance,
        electricity_costs=SITE["tariffs"]["flux"]["kwh_costs"],
        grid_charge=True,
        grid_discharge=True,
        battery=True,
        solar=True,
        color="yellow",
        verbose=True,
        saving_sessions_discharge=False,
    )
    simulate_tariff_and_store(
        name="agile incoming and fixed outgoing",
        balance=balance,
        electricity_costs=[
            {
                "start": 0,
                "end": 24,
                "import": "agile",
                "export": 0.15,
            },
        ],
        grid_charge=True,
        grid_discharge=True,
        agile_charge=True,
        battery=True,
        solar=True,
        color="blue",
        saving_sessions_discharge=True,
        verbose=False,
    )
    simulate_tariff_and_store(
        name="agile incoming and outgoing",
        balance=balance,
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
        grid_discharge=True,
        agile_charge=True,
        battery=True,
        solar=True,
        color="blue",
        saving_sessions_discharge=False,
        verbose=False,
    )


    simulate_tariff_and_store(
        name="flexible no solar no batteries",
        balance=balance,
        gas_hot_water=True,
        verbose=False,
        battery=False,
        solar=False,
    )

    simulate_tariff_and_store(
        name="flux",
        balance=balance,
        electricity_costs=SITE["tariffs"]["flux"]["kwh_costs"],
        grid_charge=True,
        battery=True,
        solar=True,
        color="green",
        saving_sessions_discharge=True,
    )
    simulate_tariff_and_store(
        name="winter agile",
        balance=balance,
        electricity_costs=SITE["tariffs"]["flux"]["kwh_costs"],
        winter_agile_import=True,
        grid_charge=True,
        agile_charge=True,
        battery=True,
        solar=True,
        color="pink",
        saving_sessions_discharge=True,
    )
    return RUN_ARCHIVE
RUN_ARCHIVE = memoize('simulations.pickle', run_simulations, max_age_days=0)

plot_simulations(RUN_ARCHIVE)
