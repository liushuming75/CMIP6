# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:25:23 2023

@author: ZLiu
"""

from plotnine import *

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as mcolors

import cmaps

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

filePath = r"D:\DATA\20221113-CMIP6T\245\ts_Amon_BCC-CSM2-MR_ssp245.nc"
data = xr.open_dataset(filePath)
lon = data['lon'][:]    #读取气象数据中的精度（lon）数据
lat = data['lat'][:]   #读取气象数据中的纬度（lat）数据
lons, lats = np.meshgrid(lon, lat)
Ts = data['ts']

print(Ts)

'''
计算未来2015-2100  全球年均温 (格点)
'''
T_year = Ts.groupby('time.year').mean()
print(T_year)

'''
计算未来2015-2100  全球总年均温
'''
T_year_mean = np.zeros(T_year.shape[0])
for time in range(0, T_year.shape[0]):
    T_year_mean[time] = np.nanmean(T_year[time,:,:])
print(T_year_mean)

'''
可视化
'''
year = pd.Series(np.arange(2015, 2101))
df = pd.DataFrame(
    {
        "Year": year,
        "Ts":T_year_mean,
    }
)

plot1 = (
    ggplot()
    + theme_matplotlib()
    + geom_line(
        df,
        aes(x="Year", y="Ts"),
        size=0.5,
        color = 'red',
#         linetype="dashed",
    )
    + geom_point(
        df,
        aes(x="Year", y="Ts"),
        size=2,
        fill='red',
        color='red',
    )
    + stat_smooth(
        df,
        aes(x="Year", y="Ts"),
        method="ols",
        color="k",
        linetype="dashed",
    )

    + scale_y_continuous(breaks=np.arange(279, 283, 1), limits=(279, 282),expand=(0,0))
    + scale_x_continuous(
        name="Year",
        breaks=np.arange(2015, 2100, 10),
        limits=(2015, 2100),
#         expand=(0,1)
    )
    + theme(
        # 设置字体 text是总的
        text=element_text(family="serif", size=13),
        axis_title_x=element_text(vjust=1, size=15),
        axis_title_y=element_text(vjust=0, size=15),
        axis_text_x=element_text(size=13, angle=0, color="black", vjust=1),
        axis_text_y=element_text(size=13, angle=0, color="black", vjust=1),
        axis_ticks_length_minor=0,
        figure_size=(8, 3),
        legend_position=(0.65, 0.25),
        legend_background=element_blank(),
        legend_title=element_blank(),
        legend_direction="horizontal",
    )
)
print(plot1)
ggsave(
    plot1,
    filename=r"C:\Users\ZLiu\Desktop\3.png",
    dpi=600,
    format="png",
    width=8,
    height=3,
)

'''
空间可视化
'''
T_year_mean1 = np.nanmean(T_year,axis=(0))
print(T_year_mean1)

#  mask  海洋  荒漠  农田
desert = r"D:\NC\Data\drought-nirv\GIMMS-NIRV-V1\CMIP\SPEI\SSP5.85\BCC-CSM2-MR\DESERT-mask.nc"
desertfile = xr.open_dataset(desert).variables["layer"].values


fig = plt.figure(figsize=(8, 6))
extent = (0, 1, 0, 1)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

ax = plt.subplot(
    projection=ccrs.PlateCarree(central_longitude=0),
)
ax.coastlines(lw=0.4)
ax.set_global()
picture = ax.contourf(
    lons,
    lats,
    T_year_mean1[:, :]*desertfile,
    cmap=cmaps.GMT_polar,
    transform=ccrs.PlateCarree(),
    extend="both",
)

ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
plt.xticks(fontproperties="Times New Roman", size=13)
plt.yticks(fontproperties="Times New Roman", size=13)
ax.set_yticks(np.arange(-90, 90, 30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))  # 经度0度不加东西
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([-179.9, 180, -90, 90], crs=ccrs.PlateCarree())

ax.tick_params(
    which="major",
    direction="out",
    length=10,
    width=0.99,
    pad=0.2,
    bottom=True,
    left=True,
    right=False,
    top=False,
)

plt.colorbar(
    picture, 
    orientation="horizontal",
    pad=0.08,
    fraction = 0.05,
#     aspect=10,
    shrink=0.8,
    label="Average future temperature (K)",
)

plt.show()
fig.savefig(r"C:\Users\ZLiu\Desktop\4.png", dpi=600, bbox_inches="tight")

'''
空间趋势可视化
'''
import pymannkendall as mk
def get_slope_p(data):
    """20"""
    if len(data[np.isnan(data)]) >= 10:
        slope = np.nan
        p_value = np.nan
    else:
        result = mk.original_test(data)
        slope = result.slope
        p_value = result.p

    return slope, p_value


slope, p = xr.apply_ufunc(
    get_slope_p,
    T_year*desertfile,
    input_core_dims=[["year"]],
    output_core_dims=[[], []],
    vectorize=True,
)

fig = plt.figure(figsize=(8, 6))
extent = (0, 1, 0, 1)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 13

ax = plt.subplot(
    projection=ccrs.PlateCarree(central_longitude=0),
)
ax.coastlines(lw=0.4)
ax.set_global()
picture = ax.contourf(
    lons,
    lats,
    slope[:, :]*desertfile,
    cmap=cmaps.MPL_hot_r,
    transform=ccrs.PlateCarree(),
    extend="both",
)

ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
plt.xticks(fontproperties="Times New Roman", size=13)
plt.yticks(fontproperties="Times New Roman", size=13)
ax.set_yticks(np.arange(-90, 90, 30), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))  # 经度0度不加东西
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.set_extent([-179.9, 180, -90, 90], crs=ccrs.PlateCarree())

ax.tick_params(
    which="major",
    direction="out",
    length=10,
    width=0.99,
    pad=0.2,
    bottom=True,
    left=True,
    right=False,
    top=False,
)

plt.colorbar(
    picture, 
    orientation="horizontal",
    pad=0.08,
    fraction = 0.05,
#     aspect=10,
    shrink=0.8,
    label="Average future temperature trend (K)",
)

plt.show()
fig.savefig(r"C:\Users\ZLiu\Desktop\5.png", dpi=600, bbox_inches="tight")