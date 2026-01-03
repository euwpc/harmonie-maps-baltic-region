import requests
import xml.etree.ElementTree as ET
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
import datetime
import os

matplotlib.use('Agg')

# --- Helper to parse QML color ramp (DISCRETE) ---
def parse_qml_colormap(qml_file):
    tree = ET.parse(qml_file)
    root = tree.getroot()
    items = []
    for item in root.findall(".//colorrampshader/item"):
        value = float(item.get('value'))
        color_hex = item.get('color').lstrip('#')
        r = int(color_hex[0:2], 16) / 255.0
        g = int(color_hex[2:4], 16) / 255.0
        b = int(color_hex[4:6], 16) / 255.0
        items.append((value, (r, g, b, 1.0)))
    items.sort(key=lambda x: x[0])
    levels = [i[0] for i in items]
    colors = [i[1] for i in items]
    return ListedColormap(colors), levels

def make_discrete_norm(levels, cmap):
    return BoundaryNorm(levels, cmap.N, clip=True)

# --- Helper for safe spatial subsetting ---
def spatial_subset(data, lon_min, lon_max, lat_min, lat_max):
    lat = data.lat
    if lat[0] < lat[-1]:
        lat_slice = slice(lat_min, lat_max)
    else:
        lat_slice = slice(lat_max, lat_min)

    return data.sel(
        lon=slice(lon_min, lon_max),
        lat=lat_slice
    )

# --- Step 1: Latest model run ---
wfs_url = "https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature&storedquery_id=fmi::forecast::harmonie::surface::grid"
response = requests.get(wfs_url, timeout=60)
response.raise_for_status()
tree = ET.fromstring(response.content)

ns = {'gml': 'http://www.opengis.net/gml/3.2', 'omso': 'http://inspire.ec.europa.eu/schemas/omso/3.0'}
origintimes = [e.text for e in tree.findall('.//omso:phenomenonTime//gml:beginPosition', ns)] or \
              [e.text for e in tree.findall('.//gml:beginPosition', ns)]
latest_origintime = max(origintimes)
run_time_str = datetime.datetime.strptime(
    latest_origintime, "%Y-%m-%dT%H:%M:%SZ"
).strftime("%Y-%m-%d %H:%M UTC")

# --- Step 2: Download data ---
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
with open("harmonie.nc", "wb") as f:
    f.write(response.content)

# --- Step 3: Load dataset ---
ds = xr.open_dataset("harmonie.nc")

temp_c = ds['air_temperature_4'] - 273.15
dewpoint_c = ds['dew_point_temperature_10'] - 273.15
pressure_hpa = ds['air_pressure_at_sea_level_1'] / 100
cape = ds['atmosphere_specific_convective_available_potential_energy_59']
windgust_ms = ds['wind_speed_of_gust_417']
precip1h_mm = ds['precipitation_amount_353'] * 3600

# --- Step 4: Load QML colormaps ---
temp_cmap, temp_levels = parse_qml_colormap("temperature_color_table_high.qml")
cape_cmap, cape_levels = parse_qml_colormap("cape_color_table.qml")
pressure_cmap, pressure_levels = parse_qml_colormap("pressure_color_table.qml")
windgust_cmap, windgust_levels = parse_qml_colormap("wind_gust_color_table.qml")
precip_cmap, precip_levels = parse_qml_colormap("precipitation_color_table.qml")

# --- Step 5: Analysis helper ---
def get_analysis(var):
    if 'time' in var.dims:
        return var.isel(time=0)
    if 'time_h' in var.dims:
        return var.isel(time_h=0)
    return var

# --- Step 6: Region ---
extent = [19.5, 30.5, 53.5, 61.5]

variables = {
    'temperature': {
        'var': temp_c,
        'cmap': temp_cmap,
        'levels': temp_levels,
        'norm': make_discrete_norm(temp_levels, temp_cmap),
        'unit': '°C',
        'title': '2m Temperature (°C)'
    },
    'dewpoint': {
        'var': dewpoint_c,
        'cmap': temp_cmap,
        'levels': temp_levels,
        'norm': make_discrete_norm(temp_levels, temp_cmap),
        'unit': '°C',
        'title': '2m Dew Point (°C)'
    },
    'pressure': {
        'var': pressure_hpa,
        'cmap': pressure_cmap,
        'levels': pressure_levels,
        'norm': make_discrete_norm(pressure_levels, pressure_cmap),
        'unit': 'hPa',
        'title': 'MSLP (hPa)'
    },
    'cape': {
        'var': cape,
        'cmap': cape_cmap,
        'levels': cape_levels,
        'norm': make_discrete_norm(cape_levels, cape_cmap),
        'unit': 'J/kg',
        'title': 'CAPE (J/kg)'
    },
    'windgust': {
        'var': windgust_ms,
        'cmap': windgust_cmap,
        'levels': windgust_levels,
        'norm': make_discrete_norm(windgust_levels, windgust_cmap),
        'unit': 'm/s',
        'title': 'Wind Gust (m/s)'
    },
    'precipitation': {
        'var': precip1h_mm,
        'cmap': precip_cmap,
        'levels': precip_levels,
        'norm': make_discrete_norm(precip_levels, precip_cmap),
        'unit': 'mm',
        'title': '1h Precipitation (mm)'
    },
}

# --- Plot ---
for key, conf in variables.items():
    data = get_analysis(conf['var'])
    cropped = spatial_subset(data, *extent)

    min_val = float(cropped.min(skipna=True))
    max_val = float(cropped.max(skipna=True))

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    cropped.plot.contourf(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=conf['cmap'],
        norm=conf['norm'],
        levels=conf['levels'],
        extend='max',
        cbar_kwargs={'label': conf['unit'], 'shrink': 0.8, 'pad': 0.05}
    )

    ax.set_extent(extent)
    ax.coastlines(resolution='10m', linewidth=1.2)
    ax.add_feature(cfeature.BORDERS, linewidth=1.2)

    plt.title(
        f"HARMONIE {conf['title']}\n"
        f"Model run: {run_time_str} | Analysis\n"
        f"Min: {min_val:.1f} {conf['unit']} | Max: {max_val:.1f} {conf['unit']}"
    )

    plt.savefig(f"{key}.png", dpi=180, bbox_inches='tight')
    plt.close()

# --- Cleanup ---
os.remove("harmonie.nc")

print("Maps generated")
