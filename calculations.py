#import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
#from jax import value_and_grad
from scipy.optimize import minimize
from scipy.stats import linregress

# from jax.scipy.optimize import minimize


def calculate_res_flow_properties(df, swin, dev, win, pi):
    df = smooth_data(df, swin, dev, win)

    # # initialize data list
    # dpq = np.zeros(len(df))  # dpq
    # time = np.arange(len(df))  # normal time
    # tmb = np.zeros(len(df))  # Material balance time
    # Q = 0

    # print(df.sample().sort_index())
    # FIND DPQ and Mat Bal Time
    df["tmb"] = calculate_material_balance_time(df["Rate"].values)
    df["dpq"] = calculate_dpq(df["Rate"].values, df["Pressure"].values, pi)
    df["tmbsqrt"] = np.sqrt(df["tmb"])
    df = df.sort_values("tmbsqrt", ascending=True)  # sort
    df = df.reset_index()  # reset indeces after drops


    return df


def calculate_material_balance_time(rates):
    Q = np.cumsum(rates)
    tmb = Q / rates
    return ffill(tmb)


# ffill along axis 1, as provided in the answer by Divakar
def ffill(arr):
    mask = np.isnan(arr) | np.isinf(arr)
    idx = np.maximum.accumulate(np.where(~mask, np.arange(arr.shape[0]), 0))
    # [1,1,0, 1]
    # [1, 2,2, 3]

    return np.where(mask, arr[idx], arr)


def calculate_dpq(rates, pressure, pi):
    dpq = (pi - pressure) / rates
    return np.where(np.isnan(dpq) | np.isinf(dpq), 0.0, dpq)

    # Sort from min to max sqrt mtb


def selected_region(df,lin_section):
    df_linear = df[df["tmbsqrt"] >= lin_section[0]]
    df_linear = df_linear[df_linear["tmbsqrt"] <= lin_section[1]]
    return df_linear

def smooth_data(df, swin, dev, win):
    # Calc Outlier Values
    df["std_rate"] = (
        df["Rate"].rolling(window=swin, center=True).std()
    )  # find Std Deviation of Rate
    df["std_rate_mean"] = (
        df["Rate"].rolling(window=swin, center=True).mean()
    )  # find mean of Rate
    df["std_pressure"] = (
        df["Pressure"].rolling(window=swin, center=True).std()
    )  # find Std Deviation of Pressure
    df["std_pressure_mean"] = (
        df["Pressure"].rolling(window=swin, center=True).mean()
    )  # find mean of Pressure

    # get rid of Outliers
    df = df.drop(df[df["Rate"] >= (dev * df["std_rate"]) + df["std_rate_mean"]].index)
    df = df.drop(
        df[df["Pressure"] >= (dev * df["std_pressure"]) + df["std_pressure_mean"]].index
    )
    df["Rate"] = (
        df["Rate"].rolling(window=win, center=True, min_periods=1).mean()
    )  # find rolling rate in window
    # lengthy = df['Rate']
    df["Pressure"] = (
        df["Pressure"].rolling(window=win, center=True, min_periods=1).mean()
    )  # find rolling pressure in wind


    return df


def print_stuff(*args):
    print(*args)


def best_split(xs, ys, early_fn, late_fn, guess=[1.0, 1.0, 1e-6, 1e-6]):
    # xs = jnp.array(xs)
    # ys = jnp.array(ys)
    # v_and_g = value_and_grad(lambda args: error(args, xs, ys, early_fn, late_fn))
    return minimize(
        error,
        guess,
        args=(xs, ys, early_fn, late_fn),
        callback=print_stuff,
        method="L-BFGS-B",
        bounds=[(1e-9, np.inf), (1.0, np.inf), (0.0, np.inf), (0.0, np.inf)],
    )


def bilinear(xs, A):
    return A * xs**1


def linear(xs, A):
    return A * xs**0.5


def pss(xs, A):
    return A * xs**0.25


def logistic(xs, A, C):
    return 1 / (1 + np.exp(-A * (xs - C)))


def mix_early_late(args, xs, early_fn, late_fn):
    A, C, A1, A2 = args
    logit = logistic(xs, A, C)
    early_fn_ans = early_fn(xs, A1)
    late_fn_ans = late_fn(xs, A2)
    return (1 - logit) * early_fn_ans + logit * late_fn_ans


def error(args, xs, ys, early_fn, late_fn):
    return ((np.log(ys) - np.log(mix_early_late(args, xs, early_fn, late_fn)))**2).sum()


def fit_flow_func(xs, ys, flow_fn, guess=[1.0]):
    err = lambda A: ((ys - flow_fn(xs, A)) ** 2).sum()
    return minimize(err, x0=np.array(guess), bounds=[(0, np.inf)])


def regress(df, lin_section):
    df_linear = df[df["tmbsqrt"] >= lin_section[0]]
    df_linear = df_linear[df_linear["tmbsqrt"] <= lin_section[1]]
    # st.write(df_linear)
    reg = linregress(df_linear["tmbsqrt"], df_linear["dpq"])
    m = reg[0]

    df_linearlog = df[df["tmbsqrt"] >= lin_section[0]]
    df_linearlog = df_linearlog[df_linearlog["tmbsqrt"] <= lin_section[1]]
    slopelog = linregress(np.log(df_linearlog["tmb"]), np.log(df_linearlog["dpq"]))
    df_linearlog = df[df["tmbsqrt"] >= lin_section[0]]
    df_linearlog = df_linearlog[df_linearlog["tmbsqrt"] <= lin_section[1]]
    slopelog = linregress(np.log(df_linearlog["tmb"]), np.log(df_linearlog["dpq"]))

    return df_linear, df_linearlog, slopelog


def test_linear():
    xs = 1
    assert linear(xs, 1) == 1


def test_logistic():
    xs = 0
    A = 1
    C = 0
    assert logistic(xs, A, C) == 0.5
    assert logistic(6, A, C) > 0.99
    assert logistic(-6, A, C) < 0.01

    assert logistic(6, A, C + 6) == 0.5
    assert logistic(12, A, C + 6) > 0.99
    assert logistic(0, A, C + 6) < 0.01


def test_mix_early_late():
    xs = 0
    A = 0
    C = 0
    A1 = 0
    A2 = 0
    args = (A, C, A1, A2)
    assert mix_early_late(args, xs, linear, bilinear) == 0
    xs = 4
    A = 0
    C = 4
    A1 = 1
    A2 = 1
    args = (A, C, A1, A2)
    answer = 0.5 * (xs**0.5) + 0.5 * (xs**1.0)
    assert mix_early_late(args, xs, linear, bilinear) == answer
    xs = 10000
    A = 1
    C = 0
    A1 = 1
    A2 = 1
    args = (A, C, A1, A2)
    answer = 0 * linear(xs, A1) + 1 * bilinear(xs, A2)
    assert abs(mix_early_late(args, xs, linear, bilinear) - answer) < 1e-3


def test_dpq():
    rates = np.array([100])
    pressure = 5000
    pi = 10000
    assert calculate_dpq(rates, pressure, pi) == 50

    rates = np.array([100])
    pressure = 10000
    pi = 10000
    assert calculate_dpq(rates, pressure, pi) == 0

    rates = np.array([0])
    pressure = 10000
    pi = 10000
    assert calculate_dpq(rates, pressure, pi) == 0


def test_calculate_material_balance_time():
    rates = np.array([1, 1, 1, 1])
    tmb = calculate_material_balance_time(rates)
    answer = np.array([1.0, 2.0, 3.0, 4.0])
    assert tmb == pytest.approx(answer)

    rates = np.array([1, 1, 0, 1])
    tmb = calculate_material_balance_time(rates)
    answer = np.array([1.0, 2.0, 2.0, 3.0])
    assert tmb == pytest.approx(answer)

    rates = np.array([1, 0, 0, 1])
    tmb = calculate_material_balance_time(rates)
    answer = np.array([1.0, 1.0, 1.0, 2.0])
    assert tmb == pytest.approx(answer)

    rates = np.array([1, 0, 1, 0, 1])
    tmb = calculate_material_balance_time(rates)
    answer = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    assert tmb == pytest.approx(answer)


def test_calculate_results():
    df = pd.read_excel("/Users/matthewwillenbring/Desktop/RTA_Data/RTA DATA 2.xlsx")
    calculate_res_flow_properties(df, 1, 1, 1, 10000)
