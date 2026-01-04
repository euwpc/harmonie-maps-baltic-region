import requests
import xml.etree.ElementTree as ET
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib
import datetime
import os
import imageio
import pandas as pd

matplotlib.use('Agg')

# --- Helper to parse QML color ramp (FIXED) ---
def parse_qml_colormap(qml_file, vmin, vmax):
    tree = ET.parse(qml_file)
    root = tree.getroot()

    values = []
    colors = []

    for item in root.findall(".//colorrampshader/item"):
        value = float(item.get('value'))
        color_hex = item.get('color').lstrip('#')

        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0

        values.append(value)
        colors.append((r, g, b, 1.0))

    values, colors = zip(*sorted(zip(values, colors)))

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(values, cmap.N, clip=True)

    return cmap, norm

# --- Step 1: Latest model run ---
wfs_url = "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&storedquery_id=fmi::forecast::harmonie::surface::grid"
response = requests.get(wfs_url, timeout=60)
response.raise_for_status()
tree = ET.fromstring(response.content)

ns = {'gml': 'http://www.opengis.net/gml/3.2', 'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0'}
origintimes = [elem.text for elem in tree.findall('.//omso:phenomenonTime//gml:beginPosition', ns)] or \
              [elem.text for elem in tree.findall('.//gml:beginPosition', ns)]
latest_origintime = max(origintimes)
run_time_str = datetime.datetime.strptime(latest_origintime, "%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M UTC")

# --- Step 2: Download with Precipitation1h ---
download_url = (
    "https://opendata.fmi.fi/download?"
    "producer=harmonie_scandinavia_surface&"
    "param=temperature,Dewpoint,Pressure,CAPE,WindGust,Precipitation1h&"
    "format=netcdf&"
    "bbox=10,53,35,71&"
    "projection=EPSG:4326"
)
response = requests.get(download_url, timeout=300)
response.raise_for_status()
nc_path = "harmonie.nc"
with open(nc_path, "wb") as f:
    f.write(response.content)

# --- Step 3: Load data ---
ds = xr.open_dataset(nc_path)

temp_c = ds['air_temperature_4'] - 273.15
dewpoint_c = ds['dew_point_temperature_10'] - 273.15
pressure_hpa = ds['air_pressure_at_sea_level_1'] / 100
cape = ds['atmosphere_specific_convective_available_potential_energy_59']
windgust_ms = ds['wind_speed_of_gust_417']
precip1h_mm = ds['precipitation_amount_353'] * 3600

# --- Step 4: Load custom colormaps ---
temp_cmap, temp_norm = parse_qml_colormap("temperature_color_table_high.qml", vmin=-40, vmax=50)
cape_cmap, cape_norm = parse_qml_colormap("cape_color_table.qml", vmin=0, vmax=5000)
pressure_cmap, pressure_norm = parse_qml_colormap("pressure_color_table.qml", vmin=870, vmax=1070)
windgust_cmap, windgust_norm = parse_qml_colormap("wind_gust_color_table.qml", vmin=0, vmax=50)
precip_cmap, precip_norm = parse_qml_colormap("precipitation_color_table.qml", vmin=0, vmax=30)

dewpoint_cmap = temp_cmap
dewpoint_norm = Normalize(vmin=-40, vmax=50)

# --- Step 5: Helper ---
def get_analysis(var):
    if 'time' in var.dims:
        return var.isel(time=0)
    elif 'time_h' in var.dims:
        return var.isel(time_h=0)
    return var

# --- Step 6: Baltic Region view ---
views = {
    'baltic': {'extent': [19.5, 30.5, 53.5, 61.5], 'suffix': ''}
}

variables = {
    'temperature': {'var': temp_c, 'cmap': temp_cmap, 'norm': temp_norm, 'unit': '°C', 'title': '2m Temperature (°C)',
                    'levels': [-40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]},
    'dewpoint':    {'var': dewpoint_c, 'cmap': dewpoint_cmap, 'norm': dewpoint_norm, 'unit': '°C', 'title': '2m Dew Point (°C)',
                    'levels': [-40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'pressure':    {'var': pressure_hpa, 'cmap': pressure_cmap, 'norm': pressure_norm, 'unit': 'hPa', 'title': 'MSLP (hPa)',
                    'levels': [890, 900, 910, 915, 920, 925, 929, 933, 938, 942, 946, 950, 954, 958, 962, 965, 968, 972, 974, 976, 978, 980, 982, 984, 986, 988, 990, 992, 994, 996, 998, 1000, 1002, 1004, 1006, 1008, 1010, 1012, 1014, 1016, 1018, 1020, 1022, 1024, 1026, 1028, 1030, 1032, 1034, 1036, 1038, 1040, 1042, 1044, 1046, 1048, 1050, 1052, 1054, 1056, 1058, 1060, 1062, 1064]},
    'cape':        {'var': cape, 'cmap': cape_cmap, 'norm': cape_norm, 'unit': 'J/kg', 'title': 'CAPE (J/kg)',
                    'levels': [0, 20, 40, 100, 200, 300, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2800, 3200, 3600, 4000, 4500, 5000]},
    'windgust':    {'var': windgust_ms, 'cmap': windgust_cmap, 'norm': windgust_norm, 'unit': 'm/s', 'title': 'Wind Gust (m/s)',
                    'levels': [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]},
    'precipitation': {'var': precip1h_mm, 'cmap': precip_cmap, 'norm': precip_norm, 'unit': 'mm', 'title': '1h Precipitation (mm)',
                      'levels': [0, 0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 30]},
}

# --- Generate Baltic region only ---
for view_key, view_conf in views.items():
    extent = view_conf['extent']
    suffix = view_conf['suffix']
    lon_min, lon_max, lat_min, lat_max = extent

    for var_key, conf in variables.items():
        data = get_analysis(conf['var'])

        # Min/max only in Baltic region (safe fallback)
        try:
            cropped_data = data.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))  # Fixed: lat_min to lat_max
            if cropped_data.size == 0:
                raise ValueError("Empty selection")
            min_val = float(cropped_data.min(skipna=True))
            max_val = float(cropped_data.max(skipna=True))
        except:
            min_val = float(data.min(skipna=True))
            max_val = float(data.max(skipna=True))

        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        # Plot full data — ensures data exists for plotting
        data.plot.contourf(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=conf['cmap'],
            norm=conf['norm'],
            levels=100,
            cbar_kwargs={'label': conf['unit'], 'shrink': 0.8, 'pad': 0.05}
        )

        if var_key == 'windgust':
            cl = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                   colors='white', linewidths=0.35,
                                   levels=conf['levels'], alpha=0.7)
            ax.clabel(cl, inline=True, fontsize=6, fmt="%d",
                      colors='black', inline_spacing=3)
        else:
            cl = data.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                   colors='black', linewidths=0.5,
                                   levels=conf['levels'])
            ax.clabel(cl, inline=True, fontsize=8, fmt="%d")

        ax.coastlines(resolution='10m', linewidth=1.2)
        ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.2, alpha=0.9)
        ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5)
        ax.set_extent(extent)

        plt.title(
            f"HARMONIE {conf['title']}\nModel run: {run_time_str} | Analysis\n"
            f"Min: {min_val:.1f} {conf['unit']} | Max: {max_val:.1f} {conf['unit']}",
            fontsize=14, pad=20
        )
        plt.savefig(f"{var_key}{suffix}.png", dpi=180, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close()

        # Animation frames
        frame_paths = []
        time_dim = 'time' if 'time' in conf['var'].dims else 'time_h'
        time_values = ds[time_dim].values

        for i in range(len(time_values)):
            if i >= 48 and (i - 48) % 3 != 0:
                continue

            fig = plt.figure(figsize=(10.24, 7.68), dpi=120)
            ax = plt.axes(projection=ccrs.PlateCarree())
            slice_data = conf['var'].isel(**{time_dim: i})

            # Min/max only in Baltic region
            try:
                slice_cropped = slice_data.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))  # Fixed: lat_min to lat_max
                if slice_cropped.size == 0:
                    raise ValueError("Empty selection")
                slice_min = float(slice_cropped.min(skipna=True))
                slice_max = float(slice_cropped.max(skipna=True))
            except:
                slice_min = float(slice_data.min(skipna=True))
                slice_max = float(slice_data.max(skipna=True))

            # Plot full slice_data — avoids "No numeric data to plot"
            slice_data.plot.contourf(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=conf['cmap'],
                norm=conf['norm'],
                levels=100
            )

            if var_key == 'windgust':
                cl = slice_data.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                             colors='white', linewidths=0.35,
                                             levels=conf['levels'], alpha=0.7)
                ax.clabel(cl, inline=True, fontsize=6, fmt="%d",
                          colors='black', inline_spacing=3)
            else:
                cl = slice_data.plot.contour(ax=ax, transform=ccrs.PlateCarree(),
                                             colors='black', linewidths=0.5,
                                             levels=conf['levels'])
                ax.clabel(cl, inline=True, fontsize=8, fmt="%d")

            ax.coastlines(resolution='10m', linewidth=1.2)
            ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=1.2, alpha=0.9)
            ax.gridlines(draw_labels=True, linewidth=0.8, color='gray', alpha=0.5)
            ax.set_extent(extent)

            valid_dt = pd.to_datetime(time_values[i]) + pd.Timedelta(hours=2)
            valid_str = valid_dt.strftime("%a %d %b %H:%M EET")

            plt.title(
                f"HARMONIE {conf['title']}\nValid: {valid_str} | +{i}h from run {run_time_str}\n"
                f"Min: {slice_min:.1f} {conf['unit']} | Max: {slice_max:.1f} {conf['unit']}",
                fontsize=13, pad=15
            )

            frame_path = f"frame_{var_key}{suffix}_{i:03d}.png"
            plt.savefig(frame_path, dpi=120, facecolor='white', pad_inches=0.3)
            plt.close()
            frame_paths.append(frame_path)

        video_path = f"{var_key}{suffix}_animation.mp4"
        with imageio.get_writer(video_path, fps=2, codec='libx264',
                                pixelformat='yuv420p', quality=8,
                                macro_block_size=16) as writer:
            for fp in frame_paths:
                writer.append_data(imageio.imread(fp))

        for fp in frame_paths:
            os.remove(fp)

if os.path.exists("harmonie.nc"):
    os.remove("harmonie.nc")

print("Baltic region maps + MP4 animations generated")
