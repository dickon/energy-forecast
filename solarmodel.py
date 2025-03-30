from forecast import parse_time, do_query, get_solar_position_index, EPOCH, SITE, tnow, get_now
import datetime, time
from asserts import assert_equal
from typing import Tuple, List
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, Response
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import threading

def populate(t, usage, actual=True):
    # if t in overridden and not actual:
    #     print('overriden', t, 'usage', usage, 'by', overridden[t])
    #     return
    pos = get_solar_position_index(t)
    if t in clear_skies or True:
        solar_pos_table.setdefault(pos, list())
        solar_pos_table[pos].append(max(0, usage))
        #print('populate', repr(t), usage, 'actual=',actual)
    if actual:
        solar_output_w[t] = max(0, usage)

def constrain_time_range(t0: datetime.datetime, t1: datetime.datetime, minute_resolution) -> Tuple[datetime.datetime, datetime.datetime]:
    if t1 is None:
        t1 = datetime.datetime.strptime(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S Z"), "%Y-%m-%d %H:%M:%S %z"
        )
    t0 -= datetime.timedelta(minutes=t0.minute % minute_resolution,
        seconds = t0.second,
        microseconds = t0.microsecond
    )
    t1 -= datetime.timedelta(minutes=t1.minute % minute_resolution,
        seconds = t1.second,
        microseconds = t1.microsecond
    )    
    return (t0, t1)

def query_powerwall( trange: Tuple[datetime.datetime, datetime.datetime], minute_resolution=30 ):
    t0, t1 = constrain_time_range(trange[0], trange[1], minute_resolution)
    prev = None
    prevt = None
    usage_wh_out = []
    for res in do_query(
        f"""
        |> filter(fn: (r) => r["_measurement"] == "energy")
        |> filter(fn: (r) => r["_field"] == "energy_exported")
        |> filter(fn: (r) => r["meter"] == "solar")
        |> window(every: {minute_resolution}m)
        |> last()
        |> duplicate(column: "_stop", as: "_time")
        |> window(every: inf) """,
        "powerwall",
        t0,
        t1,
    ):
        t = res["_time"]
        value = res["_value"]
        if prevt:
            deltat = t - prevt
            if deltat.seconds == minute_resolution*60:
                usage_wh = (value - prev) * (3600/deltat.seconds)
                usage_wh_out.append((t+datetime.timedelta(minutes=minute_resolution/2), usage_wh))
        prevt = t
        prev = value
    return usage_wh_out

def arrange_by_solar_position(data: List[Tuple[datetime.datetime, float]]):
    out = [ (get_solar_position_index(t, None, None), v) for t, v in data]
    return out

def test_constrain():
    assert_equal(
        constrain_time_range(parse_time('2025-03-28 08:00:00Z'),
        parse_time('2025-03-28 18:00:00Z'), 30), 
         (
             datetime.datetime(2025, 3, 28, 8, 0, tzinfo=datetime.timezone.utc), 
             datetime.datetime(2025, 3, 28, 18, 0, tzinfo=datetime.timezone.utc)
         ))

def test_constrain_5():
    assert_equal(
        constrain_time_range(parse_time('2023-03-28 08:15:00Z'),
        parse_time('2025-03-28 18:27:00Z'), 5),
         (
             datetime.datetime(2025, 3, 28, 8, 15, tzinfo=datetime.timezone.utc), 
             datetime.datetime(2025, 3, 28, 18, 25, tzinfo=datetime.timezone.utc)
         )
        )


TEST_GOLDEN_SAMPLE = [
    (datetime.datetime(2025, 3, 28, 9, 15, tzinfo=tzutc()), 808.0),
    (datetime.datetime(2025, 3, 28, 9, 45, tzinfo=tzutc()), 1446.0),
    (datetime.datetime(2025, 3, 28, 10, 15, tzinfo=tzutc()), 2394.0),
    (datetime.datetime(2025, 3, 28, 10, 45, tzinfo=tzutc()), 4144.0),
    (datetime.datetime(2025, 3, 28, 11, 15, tzinfo=tzutc()), 4554.0),
    (datetime.datetime(2025, 3, 28, 11, 45, tzinfo=tzutc()), 6952.0),
    (datetime.datetime(2025, 3, 28, 12, 15, tzinfo=tzutc()), 6014.0),
    (datetime.datetime(2025, 3, 28, 12, 45, tzinfo=tzutc()), 4998.0),
    (datetime.datetime(2025, 3, 28, 13, 15, tzinfo=tzutc()), 5860.0),
    (datetime.datetime(2025, 3, 28, 13, 45, tzinfo=tzutc()), 2928.0),
    (datetime.datetime(2025, 3, 28, 14, 15, tzinfo=tzutc()), 5198.0),
    (datetime.datetime(2025, 3, 28, 14, 45, tzinfo=tzutc()), 6622.0),
    (datetime.datetime(2025, 3, 28, 15, 15, tzinfo=tzutc()), 5302.0),
    (datetime.datetime(2025, 3, 28, 15, 45, tzinfo=tzutc()), 6562.0),
    (datetime.datetime(2025, 3, 28, 16, 15, tzinfo=tzutc()), 5114.0),
    (datetime.datetime(2025, 3, 28, 16, 45, tzinfo=tzutc()), 4066.0),
    (datetime.datetime(2025, 3, 28, 17, 15, tzinfo=tzutc()), 1984.0),
    (datetime.datetime(2025, 3, 28, 17, 45, tzinfo=tzutc()), 780.0),
    (datetime.datetime(2025, 3, 28, 18, 15, tzinfo=tzutc()), 228.0),
]

TEST_POS = [
    ((32, 133), 808.0),
    ((35, 141), 1446.0),
    ((37, 149), 2394.0),
    ((39, 159), 4144.0),
    ((41, 168), 4554.0),
    ((41, 178), 6952.0),
    ((41, 188), 6014.0),
    ((40, 198), 4998.0),
    ((38, 207), 5860.0),
    ((36, 216), 2928.0),
    ((33, 224), 5198.0),
    ((29, 232), 6622.0),
    ((26, 239), 5302.0),
    ((22, 246), 6562.0),
    ((17, 253), 5114.0),
    ((13, 259), 4066.0),
    ((8, 265), 1984.0),
    ((4, 271), 780.0),
    ((-1, 277), 228.0),
]


def test_query_powerwall():
    assert_equal(
        query_powerwall(
            (parse_time("2025-03-28 08:15:00Z"), parse_time("2025-03-28 18:12:00Z")),
        ),
       TEST_GOLDEN_SAMPLE
    )


def test_arranage_by_solar_position():
    assert_equal(arrange_by_solar_position(TEST_GOLDEN_SAMPLE), TEST_POS)    

def cap_values(limit, data):
    return [ (k, min(v, limit)) for k,v in data]

def bin_max(data, grid0=None, altitude_resolution=1, azimuth_resolution=1):
    grid = {} if grid0 is None else grid0
    for pos, wh in data:
        pos_round = round(pos[0]/altitude_resolution)*altitude_resolution, round(pos[1]/azimuth_resolution)*azimuth_resolution
        grid[pos] = max(grid.get(pos, 0), wh)
    return grid

def plot_png(state, minute_resolution, metadata):
    altitude = [x[0][0] for x in state.items()]
    azimuth = [x[0][1] for x in state.items()]
    pow = [x[1] for x in state.items()]
    
    #p = plt.scatter(azimuth, altitude, [x / 60 for x in pow], pow)
    matplotlib.use('Agg')
    plt.margins(0)
    fig, ax = plt.subplots()
    p = plt.scatter(azimuth, altitude, 2, pow)
    ax.set_ylim([0, 70])
    ax.set_xlim((50, 300))
    ax.set_title(f"solar AC output power over {minute_resolution} minute{'s' if minute_resolution > 1 else ''} latest {metadata.get('latest')}")
    ax.set_xlabel("azimuth")
    ax.set_ylabel("altitude")
    ax.annotate( "sun", xy=get_solar_position_index(get_now()))
    #ax.colorbar()
    output = io.BytesIO()
    plt.savefig(output)
    plt.close()
    global image
    image = output.getvalue()
    print('image made size', len(image))
    return image

def read(state, minute_resolution, metadata):
    t = parse_time('2023-08-01 00:00:00Z')
    while t < tnow:
        tnext = t + datetime.timedelta(days=1)
        qdata = query_powerwall((t, tnext), minute_resolution)
        if qdata:
            peak = max([ v for _,v in qdata])
        else:
            peak = 0
        bin_max(arrange_by_solar_position(cap_values(SITE['solar']['plausible_maximum_power_w'], qdata)),state, 5, 5)
        if len(qdata) > 0:
            t= qdata[-1][0]
            metadata['latest'] = t
            
            delta = tnow - qdata[-1][0]
            sleep = 3600 if delta < datetime.timedelta(days=1) else 1
        else:
            delta = tnext - tnow
            t = tnext
            sleep = 0.1

        print(f'{threading.get_ident()} {t} delta {delta} recs {len(qdata)} peak {int(peak)}Wh populated {len(state.keys())} now sleep {sleep}')
        plot_start = time.time()
        plot_png(state, minute_resolution, metadata)
        print(f'\tplot in {time.time() - plot_start}')
        time.sleep(sleep)

def serve():
    global image
    metadata = {}
    app = Flask(__name__)
    minute_resolution = 1
    state = {}
    data_thread = threading.Thread(target=read, args=(state,minute_resolution, metadata))
    data_thread.start()
    image = None
    @app.route('/')
    def show_png():        
        #image = plot_png(state, minute_resolution, metadata)
        print('image size', len(image))
        return Response(image, mimetype='image/png')
    app.run(port=5005)

if __name__ == '__main__':
    serve()
