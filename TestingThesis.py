

#import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
#import streamlit.components.v1 as components
from matplotlib.offsetbox import AnchoredText

from calculations import (
    best_split,
    bilinear,
    calculate_res_flow_properties,
    error,
    fit_flow_func,
    linear,
    logistic,
    mix_early_late,
    pss,
    regress,
    selected_region,
)

st.set_page_config(page_title="RTA Uncertainty", layout="wide")
col0, col00 =st.columns(2)

with col0:
    st.title("Rate Transient Analysis")
with col00:
    st.header("By: Matthew Willenbring")


with st.sidebar:
    pi = st.text_input("Initial Pressure [psi]")
    if pi == "":
        pi = 10000
    else:
        pi = int(pi)

    boi = 0.0
    ki = 0.0
    cti = 0.0
    pori = 0.0
    si = 0.0
    etli = 0
    fni = 0
    hi = 0
    µoi = 0.0
    xfi = 0.0

    bo = st.number_input("Bo [RB/STB]", value=boi, step=0.01)
    k = float(st.text_input("Permiability [md]", value="0.001"))
    ct = float(st.text_input("Total Compressability [psi-1]", value="1e-6"))
    µo = st.number_input("Oil Viscocity", value=µoi, step=0.1)
    xf = st.number_input("Fracture Half Length", value=xfi)
    h = st.number_input("Height of Reservoir [ft]", value=hi)
    por = st.number_input("Porosity [%]", value=pori, step=0.01)
    #skin = st.number_input("Skin", value=si, step=0.01)
    #etl = st.number_input("Length of Lateral [ft]", value=etli)
    #fn = st.number_input("Number of fractures", value=fni)


data = st.file_uploader(
    "Submit Excel:     pressure [psi], rate [bbl/day]", type=["xls", "xlsx"]
)

#if st.button("Use Example Data"):
#    data = "RTA DATA 2.xlsx"


if data is not None:
    pass
else:
    data = 'RTA DATA 2.xlsx'

# Get slider Data for Cleaning
swin = st.slider(
    "Outlier: Window Size", min_value=2, max_value=20, value=11
)  # Window size slider for outlier
dev = st.slider(
    "Outlier: Standard Deviations",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.5,
)  # Slider for number of deviations





# Smoothing by Mean
win = st.slider("Smoothing: Window Size", min_value=1, max_value=50, value=12)

df = calculate_res_flow_properties(pd.read_excel(data), swin, dev, win, pi)



    # Plotting Rate and Pressure
figt, axt = plt.subplots(figsize=(8, 4))  # create figure
axz = axt.twinx()  # create dual axis graph

Pressure_Color = "#69b3a2"  # Color Codes for Graph
Rate_Color = "#3399e6"

axt.grid()
figt.suptitle("Pressure and Rate")
axt.set_xlabel("Days")
axt.plot(df["Pressure"], color=Pressure_Color)
axz.plot(df["Rate"], color=Rate_Color)
axz.tick_params(axis="y", labelcolor=Rate_Color)
axt.tick_params(axis="y", labelcolor=Pressure_Color)
axt.set_ylabel("Pressure [psi]", color=Pressure_Color)
axz.set_ylabel("Rate [Bbls]", color=Rate_Color)

st.pyplot(figt)  # plot pressure and rate




# create subplots
fig, x = plt.subplots()
figz, z = plt.subplots()
figy, y = plt.subplots()



# slider for flow type
lin_section = st.slider(
    "Flow Region",
    min_value=0.0,
    max_value=df["tmbsqrt"].max(),
    value=(0.0, float(df["tmbsqrt"].max())),
    step=0.1,
)


lin_data = df.loc[(df["tmbsqrt"] >= lin_section[0]) & (df["tmbsqrt"] <= lin_section[1])]
post_lin_data = df.loc[(df["tmbsqrt"] > lin_section[1])]
pre_lin_data =  df.loc[(df["tmbsqrt"] < lin_section[0])]

x.scatter(lin_data["tmbsqrt"], lin_data["dpq"], color="green")
x.scatter(post_lin_data["tmbsqrt"], post_lin_data["dpq"], color="red")
x.scatter(pre_lin_data["tmbsqrt"], pre_lin_data["dpq"], color="red")

#y.scatter(df["tmb"], df["dpq"], color="green")
y.scatter(lin_data["tmb"], lin_data["dpq"], color="green")
y.scatter(post_lin_data["tmb"], post_lin_data["dpq"], color="red")
y.scatter(pre_lin_data["tmb"], pre_lin_data["dpq"], color="red")

df = selected_region(df,lin_section)

# create columns
col1, col2 = st.columns(2)

df_linear, df_linearlog, slopelog = regress(df, lin_section)
# st.write(dataframe)
with col1:
    early_flow_selection = st.selectbox(
        "Early Flow Function", ("linear", "bilinear", "pss"), 1
    )
with col2:
    late_flow_selection = st.selectbox(
        "Late Flow Function", ("linear", "bilinear", "pss"), 0
    )

results_of_linear = fit_flow_func(df["tmb"], df["dpq"], linear)
results_of_bilinear = fit_flow_func(df["tmb"], df["dpq"], bilinear)
results_of_pss = fit_flow_func(df["tmb"], df["dpq"], pss)
function_parts = {
    "linear": {"fn": linear, "guess": results_of_linear.x[0]},
    "bilinear": {
        "fn": bilinear,
        "guess": results_of_bilinear.x[0],
    },
    "pss": {"fn": pss, "guess": results_of_pss.x[0]},
}

early_fn_def = function_parts[early_flow_selection]
late_fn_def = function_parts[late_flow_selection]

results_of_best_split = best_split(
    df["tmb"],
    df["dpq"],
    early_fn_def["fn"],
    late_fn_def["fn"],
    guess=[1.0, 1.0, early_fn_def["guess"], late_fn_def["guess"]],
)
df["mixed_curve"] = mix_early_late(
    results_of_best_split.x, df["tmb"], early_fn_def["fn"], late_fn_def["fn"]
)

#st.write(results_of_best_split)

# Title and Description
#with st.expander("Smoothing Calculations: ", expanded=False):
#    st.header("Smoothing Data")
#    st.write(
#        "Based on the value selected, RTA Solver with do a moving window average of y values with the selected number of points in window"
#    )





# color coding region selected


x.scatter(df["tmbsqrt"], df["mixed_curve"], color="grey", alpha=0.2)
y.scatter(df["tmb"], df["mixed_curve"], color="grey", alpha=0.2)
# x.plot(xrange, tmbsqr_spline(xrange))
# z.plot(xrange,der2(xrange))

# linear graph parameters
x.grid(True)
x.set_title("DPQ")
x.set_xlabel("Sqrt Material Balance Time")
x.set_ylabel("dp/q")

# log plot parameters

y.grid(True)
y.set_xscale("log")
y.set_yscale("log")
y.set_title("LOG LOG dp/q Plot")
y.set_xlabel("Material Balance Time")
y.set_ylabel("dp/q")

#anchored_text = AnchoredText(
#    "Slope of LOG Selection: " + str(round(slopelog[0], 5)),
#    loc=2,
#    prop=dict(fontweight="bold"),
#)
#y.add_artist(anchored_text)



# plot graphs in columns
with col1:
    st.pyplot(fig)
with col2:
    st.pyplot(figy)


z.set_title("Logistic Function")
anchored_text = AnchoredText(
    "Early Flow Regime: " + str(early_flow_selection),
    loc=2,
    prop=dict(fontweight="bold")
)
z.add_artist(anchored_text)

anchored_text = AnchoredText(
    "Late Flow Regime: " + str(late_flow_selection),
    loc=1,
    prop=dict(fontweight="bold")
)
z.add_artist(anchored_text)

z.set_ylim(0,1)
z.grid()
z.set_ylabel("Early Flow Percent")
z.set_xlabel("Material Balance Time")
z.scatter(
    df["tmb"],
    logistic(df["tmb"], results_of_best_split.x[0], results_of_best_split.x[1]),
)
with col1:
    st.pyplot(figz)
with col2:
    st.subheader("     " + str(early_flow_selection).capitalize() + " Flow Slope: " + str(round(results_of_best_split.x[2],2)))
    st.subheader("     " + str(late_flow_selection).capitalize() + " Flow Slope: " + str(round(results_of_best_split.x[3],2)))

#with col1:
 #   z.set_ylim(-100, 100)
 #   st.pyplot(figz)

# Finding Fracture Surface Area
with col2:
    if st.button("Find Fracture Surface Area"):
        if early_flow_selection == "linear" or late_flow_selection == "linear":
            variables = [bo, k, ct, µo, por]
            if all(var>0 for var in variables):
                if early_flow_selection == "linear":
                    m = results_of_best_split.x[2]
                else:
                    m = results_of_best_split.x[3]
                #st.write(m)
                A = (4 * 19.927 * (bo / m) * ((µo / (por * ct))))**0.5 / ((k)**0.5)
                st.write(
                    "Surface Area of Fractures:  "
                    + str(("{:,}".format(round(A))))
                    + " ft²"
                )
            else:
                st.warning("Check input data: Bo, k, Ct, µo, and porosity needed")
        else:
            st.warning("Use a Linear Flow Period to calculate Fracture Surface Area")


with col2:
    if st.button("Find Fracture Conductivity"):
        if early_flow_selection == "bilinear" or late_flow_selection == "bilinear":
            variables = [bo, k, ct, µo, por, h]
            if all(var>0 for var in variables):
                if early_flow_selection == "bilinear":
                    m = results_of_best_split.x[2]
                else:
                    m = results_of_best_split.x[3]
                #st.write(m)
                kwf = (((44.13 * µo * bo)/(m * h)) * ((k/(µo * por * ct))**0.25))**2
                st.write(
                    "Fracture Conductivity:  "
                    + str(("{:,}".format(round(kwf))))
                    + " md⋅ft"
                )
            else:
                st.warning("Check input data: Bo, k, Ct, µo, and porosity needed")
        else:
            st.warning("Use a Linear Flow Period to calculate Fracture Surface Area")

