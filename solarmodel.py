from forecast import parse_time, do_query, get_solar_position_index, EPOCH
import datetime
from asserts import assert_equal
from typing import Tuple
from dateutil.tz import tzutc

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

def constrain_time_range(t0: datetime.datetime, t1: datetime.datetime, minute_resolution=30) -> Tuple[datetime.datetime, datetime.datetime]:
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
        """
        |> filter(fn: (r) => r["_measurement"] == "energy")
        |> filter(fn: (r) => r["_field"] == "energy_exported")
        |> filter(fn: (r) => r["meter"] == "solar")
        |> window(every: 30m)
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
                usage_wh_out.append((t, usage_wh))
        prevt = t
        prev = value
    return usage_wh_out

def test_constrain():
    assert_equal(
        constrain_time_range(parse_time('2025-03-28 08:00:00Z'),
        parse_time('2025-03-28 18:00:00Z')), 
         (
             datetime.datetime(2025, 3, 28, 8, 0, tzinfo=datetime.timezone.utc), 
             datetime.datetime(2025, 3, 28, 18, 0, tzinfo=datetime.timezone.utc)
         ))

def test_constrain_5():
    assert_equal(
        constrain_time_range(parse_time('2025-03-28 08:15:00Z'),
        parse_time('2025-03-28 18:27:00Z'), 5),
         (
             datetime.datetime(2025, 3, 28, 8, 15, tzinfo=datetime.timezone.utc), 
             datetime.datetime(2025, 3, 28, 18, 25, tzinfo=datetime.timezone.utc)
         )
        )


TEST_GOLDEN_SAMPLE = [
            (datetime.datetime(2025, 3, 28, 9, 0, tzinfo=tzutc()), 808.0),
            (datetime.datetime(2025, 3, 28, 9, 30, tzinfo=tzutc()), 1446.0),
            (datetime.datetime(2025, 3, 28, 10, 0, tzinfo=tzutc()), 2394.0),
            (datetime.datetime(2025, 3, 28, 10, 30, tzinfo=tzutc()), 4144.0),
            (datetime.datetime(2025, 3, 28, 11, 0, tzinfo=tzutc()), 4554.0),
            (datetime.datetime(2025, 3, 28, 11, 30, tzinfo=tzutc()), 6952.0),
            (datetime.datetime(2025, 3, 28, 12, 0, tzinfo=tzutc()), 6014.0),
            (datetime.datetime(2025, 3, 28, 12, 30, tzinfo=tzutc()), 4998.0),
            (datetime.datetime(2025, 3, 28, 13, 0, tzinfo=tzutc()), 5860.0),
            (datetime.datetime(2025, 3, 28, 13, 30, tzinfo=tzutc()), 2928.0),
            (datetime.datetime(2025, 3, 28, 14, 0, tzinfo=tzutc()), 5198.0),
            (datetime.datetime(2025, 3, 28, 14, 30, tzinfo=tzutc()), 6622.0),
            (datetime.datetime(2025, 3, 28, 15, 0, tzinfo=tzutc()), 5302.0),
            (datetime.datetime(2025, 3, 28, 15, 30, tzinfo=tzutc()), 6562.0),
            (datetime.datetime(2025, 3, 28, 16, 0, tzinfo=tzutc()), 5114.0),
            (datetime.datetime(2025, 3, 28, 16, 30, tzinfo=tzutc()), 4066.0),
            (datetime.datetime(2025, 3, 28, 17, 0, tzinfo=tzutc()), 1984.0),
            (datetime.datetime(2025, 3, 28, 17, 30, tzinfo=tzutc()), 780.0),
            (datetime.datetime(2025, 3, 28, 18, 0, tzinfo=tzutc()), 228.0),
        ]   
def test_query_powerwall():
    assert_equal(
        query_powerwall(
            (parse_time("2025-03-28 08:15:00Z"), parse_time("2025-03-28 18:12:00Z")),
        ),
       TEST_GOLDEN_SAMPLE
    )


if __name__ == '__main__':
    test_query_powerwall()
