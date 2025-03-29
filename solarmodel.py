from forecast import parse_time, do_query, get_solar_position_index, EPOCH
import datetime
from asserts import assert_equal
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


def query_powerwall(t0, t1=None):
    if t1 is None:
        t1 = datetime.datetime.strptime(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S Z"), "%Y-%m-%d %H:%M:%S %z"
        )
    t0 = t0.replace(minute=0, second=0, microsecond=0)
    t1 = t1.replace(minute=0, second=0, microsecond=0)
    prev = None
    prevt = None
    usage_wh_total = 0
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
            #print('powerwall', deltat, res)
            if deltat.seconds == 1800:
                usage_wh = (value - prev) * (3600/deltat.seconds)
                usage_wh_total += usage_wh
        prevt = t
        prev = value
    return usage_wh_total

def test_query_powerwall():
    assert_equal(query_powerwall(
        parse_time('2025-03-28 08:00:00Z'),
        parse_time('2025-03-28 18:00:00Z'),
    ), 75954)

if __name__ == '__main__':
    test_query_powerwall()
