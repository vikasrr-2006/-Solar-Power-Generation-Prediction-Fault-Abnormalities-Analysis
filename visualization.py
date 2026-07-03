import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_daily_dc_power(df):
    daily_dc = df.groupby('DATE')['DC_Power'].agg('sum')
    ax = daily_dc.sort_values(ascending=False).plot.bar(
        figsize=(17, 5), legend=True, color='red'
    )
    plt.title('Daily DC Power')
    plt.show()
    return daily_dc


def compare_best_worst_days(df, best_date, worst_date):
    plt.figure(figsize=(16, 16))

    plt.subplot(411)
    sns.lineplot(
        df[df["DATE_STRING"].isin([best_date])].DATE_TIME,
        df[df["DATE_STRING"].isin([best_date])]["DC_Power"],
        label="DC_Power_Best", color='green'
    )
    plt.title("DC Power Generation: {}".format(best_date))

    plt.subplot(412)
    sns.lineplot(
        df[df["DATE_STRING"].isin([best_date])].DATE_TIME,
        df[df["DATE_STRING"].isin([best_date])].IRRADIATION,
        label="Irradiation_Best", color='green'
    )
    plt.title("Irradiation: {}".format(best_date))

    plt.subplot(413)
    sns.lineplot(
        df[df["DATE_STRING"].isin([best_date])].DATE_TIME,
        df[df["DATE_STRING"].isin([best_date])].AMBIENT_TEMPERATURE,
        label="Ambient_Temperature_Best", color='green'
    )
    sns.lineplot(
        df[df["DATE_STRING"].isin([best_date])].DATE_TIME,
        df[df["DATE_STRING"].isin([best_date])].MODULE_TEMPERATURE,
        label="Module_Temperature_Best", color='blue'
    )
    plt.title("Module Temperature & Ambient Temperature: {}".format(best_date))

    plt.tight_layout()
    plt.show()


def calculate_inverter_efficiency(df):
    solar_dc_power = df[df['DC_Power'] > 0]['DC_Power'].values
    solar_ac_power = df[df['AC_POWER'] > 0]['AC_POWER'].values
    efficiency = (np.max(solar_ac_power) / np.max(solar_dc_power)) * 100
    return efficiency
