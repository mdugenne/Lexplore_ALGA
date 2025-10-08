## Objective: This script generates summary data and figures from the database available at https://www.alplakes.eawag.ch/
# Load modules and functions required for image processing
from functools import partial

import shapely.plotting
import warnings

from matplotlib.font_manager import json_dump

warnings.filterwarnings(action='ignore',category=SyntaxWarning)
import cartopy.crs
import pandas as pd
import scipy.spatial.distance
import xarray

try:
    from funcs_image_processing import *

except:
    from scripts.funcs_image_processing import *

import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
# Maps
import folium
#map=folium.Map(location=[46.5002275,6.660835277777778],tiles='Cartodb Positron',zoom_start=4)
#map.save('{}/figures/alplakes/map_backdrop.png'.format(str(path_to_git)))
import io
from PIL import Image

#img_data = map._to_png(5)
#img = Image.open(io.BytesIO(img_data))
#img.save('{}/figures/alplakes/map_backdrop.png'.format(str(path_to_git)))


# Load functions from https://github.com/eawag-surface-waters-research/alplakes-simulations/blob/master/src/functions.py
def get_closest_index(value, array):
    array = np.asarray(array)
    sorted_array = np.sort(array)
    if len(array) == 0:
        raise ValueError("Array must be longer than len(0) to find index of value")
    elif len(array) == 1:
        return 0
    if value > (2 * sorted_array[-1] - sorted_array[-2]):
        raise ValueError("Value {} greater than max available ({})".format(value, sorted_array[-1]))
    elif value < (2 * sorted_array[0] - sorted_array[-1]):
        raise ValueError("Value {} less than min available ({})".format(value, sorted_array[0]))
    return (np.abs(array - value)).argmin()
from datetime import datetime
def convert_time(time):
    return datetime.fromtimestamp(time + (datetime(2008, 3, 1).replace(tzinfo=timezone.utc) - datetime(1970, 1, 1).replace(tzinfo=timezone.utc)).total_seconds()).strftime('%Y-%m-%d %H:%M:%S')


def rotate_velocity(u, v, alpha):
    u = np.asarray(u).astype(np.float64)
    v = np.asarray(v).astype(np.float64)
    alpha = np.asarray(alpha).astype(np.float64)

    u[u == -999.0] = np.nan
    v[v == -999.0] = np.nan
    alpha[alpha == 0.0] = np.nan

    alpha = np.radians(alpha)
    u_n = u * np.cos(alpha) - v * np.sin(alpha)
    v_e = v * np.cos(alpha) + u * np.sin(alpha)

    return u_n, v_e

import seaborn as sns
palette_oxygen=list(reversed(sns.color_palette("inferno",15).as_hex()))
# Data processing modules
import os
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from pyproj import Transformer
transformer_ch_to_ws = Transformer.from_crs("EPSG:21781", "EPSG:4326", always_xy=True) #CH1903
transformer_ws_to_ch= Transformer.from_crs( "EPSG:4326","EPSG:21781", always_xy=True) #CH1903
import xarray as xr
from natsort import natsorted
import rasterio
import rasterio.plot
import geopandas as gpd
from geocube.api.core import make_geocube
from geocube.rasterize import rasterize_points_griddata # Different choices here
from shapely.geometry import Polygon, Point
r'''
bathy_files=natsorted(list(Path(r"D:\Users\dugenne\\Downloads\swissbathy3d_lacleman_2056_5728.xyz").expanduser().glob('*.xyz')))
df_bathy=pd.concat(map(lambda file: pd.read_table(file,sep=' '),bathy_files))
df_bathy.columns=['x','y','z']
transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True) #CH1995
df_bathy.x,df_bathy.y=transformer.transform(df_bathy.x,df_bathy.y)
df_bathy.z=df_bathy.z-df_bathy.z.min()
df_bathy.z=df_bathy.z-df_bathy.z.max()


#Rasterize
# GeoDataFrame
gf = gpd.GeoDataFrame(df_bathy, geometry=gpd.points_from_xy(df_bathy.x, df_bathy.y),  crs=4326)
# Rasterize Points to Gridded Xarray Dataset
geo_grid = make_geocube(
    vector_data=gf,
    measurements=['z'],
    resolution=0.002, # degrees
    rasterize_function=rasterize_points_griddata,
)

geo_grid = geo_grid.assign_attrs(units='meters',description='bathymetry')
geo_grid.to_netcdf(path_to_git / 'data' /'bathymetry_geneva.nc',engine='h5netcdf')
plt.figure()
geo_grid.z.plot(cmap='twilight')
plt.show()
'''
nc_bathy = xr.open_dataset(path_to_git / 'data' /'bathymetry_geneva.nc')
plt.figure()
plt.pcolormesh(np.array((-1*nc_bathy.z))[::-1,:])
plt.pcolormesh(np.array((nc_bathy.z<-15))[::-1,:])
plt.show()

fig, ax = plt.subplots(1)
cs=plt.contour(nc_bathy.x,nc_bathy.y[::-1],nc_bathy.z[::-1,:],levels=[-13,-10],alpha=0.5)
polygon_bathy=shapely.intersection(Polygon(cs.allsegs[0][0]),Polygon(cs.allsegs[1][0]))
from shapely.plotting import plot_polygon
#plot_polygon(polygon_bathy)

df_raster=nc_bathy.to_dataframe().reset_index()
from urllib.request import urlopen
def tiff_rasterize(tiff,time_index,depth_index=0,grid_resolution=0.002):

def download_sentinel_data(year='2024',month='06',path_output=path_to_git / 'data' / 'datafiles' / 'alplakes' / 'sentinel'):
    df_metadata=pd.DataFrame(json.loads(urlopen('https://alplakes-api.eawag.ch/remotesensing/products/geneva/sentinel3/chla').read()))
    df_metadata['datetime']=pd.to_datetime(df_metadata.datetime,format='%Y%m%dT%H%M%S')
    df_metadata['percentage_valid_pixels']=df_metadata['valid_pixels'].str[0:-1].astype(float)
    df_metadata=df_metadata[(df_metadata.datetime>=np.datetime64(datetime(int(year),int(month),1))) & (df_metadata.datetime<np.datetime64(datetime(int(year),int(month)+1,1)))]
    df_metadata=df_metadata.loc[df_metadata.groupby(['datetime']).percentage_valid_pixels.idxmax()]
    df_metadata['week']=df_metadata['datetime'].dt.strftime('%Y-%m')+' '+np.mod(df_metadata['datetime'].dt.strftime('%W').astype(int),7).astype(str).str.zfill(2)
    with requests.Session() as session:
        for week in df_metadata.week.unique():
            path_weekly_files=[]
            file_url = df_metadata.loc[df_metadata.week==week, 'url']
            for file in file_url:
                path_to_file = Path(path_output).expanduser() / str(file).split(r'/')[-1]
                path_weekly_files=path_weekly_files+[path_to_file]
                # Get chl-a estimates
                with session.get(file, stream=True, headers={'Connection': 'Keep-Alive','User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}) as rsp:
                    with open(path_to_file, "wb") as f:
                        for a_chunk in rsp.iter_content(chunk_size=131072):  # Loop over content, i.e. eventual HTTP chunks
                            f.write(a_chunk)
                # Get secchi depth estimates
                path_to_file=
                secchi_file_url=file
                with session.get(file, stream=True, headers={'Connection': 'Keep-Alive','User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}) as rsp:
                    with open(path_to_file, "wb") as f:
                        for a_chunk in rsp.iter_content(chunk_size=131072):  # Loop over content, i.e. eventual HTTP chunks
                            f.write(a_chunk)

            nc = xr.open_mfdataset(path_weekly_files,combine='nested',concat_dim='time')
            nc.coords['time']=df_metadata.loc[df_metadata.week==week, 'datetime']
            nc=nc.sel(y=slice(df_raster.y.max(),df_raster.y.min()),x=slice(df_raster.x.min(),df_raster.x.max()),band=1)
            nc=nc.mean(dim=['time'])
            fig, ax = plt.subplots()
            plot=plt.pcolormesh(nc.variables['band_data'].data[::-1])
            fig.colorbar(plot, ax=ax)

            df_gf = gpd.GeoDataFrame(,
                                     geometry=gpd.points_from_xy(df_coords_untf.dropna(subset=['lat', 'lon']).lon,
                                                                 df_coords_untf.dropna(subset=['lat', 'lon']).lat),
                                     crs=4326)
            # Rasterize Points to Gridded Xarray Dataset

            geo_grid = make_geocube(
                vector_data=df_gf,  # [df_gf.time==df_coords_untf.time.unique()[0]]
                measurements=['temperature', 'u', 'v', 'w', 'velocity_magnitude', 'bottom_stress_u', 'bottom_stress_v',
                              'bottom_stress_max', 'eddy_viscosity', 'eddy_diffusivity', 'hydrostatic_pressure',
                              'thermocline_depth', 'water_density'],  #
                datetime_measurements=['time'],
                resolution=grid_resolution,  # degrees
                rasterize_function=rasterize_points_griddata,
                group_by="time",
            )

            # Mask raster according to bathymetry
            df_polygon = geo_grid.to_dataframe().reset_index()[['x', 'y']].drop_duplicates(subset=['x', 'y'])
            points = list(map(lambda coords: Point(coords[0], coords[1]), df_polygon.values))
            df_polygon = df_polygon.assign(z=polygon_bathy.contains(points))
            masked_geo_grid = xr.merge([geo_grid, xr.Dataset.from_dataframe(df_polygon.set_index(['x', 'y']))])
            masked_geo_grid = xr.where(masked_geo_grid.z == True, masked_geo_grid, np.nan, keep_attrs=True)
            geo_grid = masked_geo_grid.transpose()

            # fig, ax = plt.subplots()
            # plot=ax.pcolormesh(geo_grid.coords['x'].data,geo_grid.coords['y'].data[::-1],geo_grid.temperature.data[0,::-1],cmap='twilight')
            # fig.colorbar(plot, ax=ax)
            # ax.set_title(pd.to_datetime(nc.variables['time'].values[time_index].tolist())[0])


def flow3D_rasterize(nc,time_index,depth_index=-1,grid_resolution=0.002):
    lon, lat = transformer_ch_to_ws.transform(nc.variables["XZ"][:].data, nc.variables["YZ"][:].data)
    np.unique(nc.variables["XZ"][:].data)
    lon[nc.variables["XZ"][:].data == 0] = np.nan
    lat[nc.variables["YZ"][:].data == 0] = np.nan
    if depth_index==-1:
        # Extract bottom depth in each grid cell
        dict_depth_index=dict(map(lambda z: (z,get_closest_index(z, np.array(nc.variables["ZK_LYR"][:]) * -1)),np.unique(nc.variables["DP0"][:].values.tolist())[1:]))
        dict_depth_index.update({np.unique(nc.variables["DP0"][:].values.tolist())[0]:1}) # selector does not allow for nan or None
        #Map effective maximum depth to corresponding z-layer index and define it as indexer
        array_depth_index=np.vectorize(dict_depth_index.get)(nc.variables["DP0"][:].values)
        depth_indexor = xr.DataArray(array_depth_index, dims=('M', 'N'))
        nc.variables["U1"].dims = ('time', 'KMAXOUT_RESTR', 'M', 'N')
        nc.variables["V1"].dims = ('time', 'KMAXOUT_RESTR', 'M', 'N')


        nc=nc.isel(KMAXOUT_RESTR=depth_indexor,KMAXOUT=depth_indexor)
        depth='bottom depth'
    else:
        nc = nc.isel(KMAXOUT_RESTR=depth_index,KMAXOUT=depth_indexor)
        depth=(np.array(nc.variables["ZK_LYR"][:]) * -1)[depth_index]


    # Extract u and v velocities and rotate
    u, v, = rotate_velocity(nc.variables["U1"][time_index, :],  # time, x, y
                            nc.variables["V1"][time_index, :],  # time, x ,y
                            np.array(nc.variables["ALFAS"][:]))
    velocity_magnitude = (u ** 2 + v ** 2) ** 0.5

    df_coords_untf = pd.DataFrame({'lon': np.tile(lon.reshape(np.prod(lon.shape)),len(time_index)), 'lat': np.tile(lat.reshape(np.prod(lon.shape)),len(time_index)),
                                   'time':np.tile(pd.to_datetime(nc.variables['time'].values[time_index].tolist()),np.prod(lon.shape)),
                                   'depth':[depth]*np.prod(lon.shape)*len(time_index),
                                   'temperature': nc.variables["R1"][time_index,0, :].data.reshape(np.prod(lon.shape)*(len(time_index))),
                                   'u': u.reshape(np.prod(lon.shape)*len(time_index)), # in meter per second
                                   'v': v.reshape(np.prod(lon.shape)*len(time_index)), # in meter per second
                                   'w':nc.variables["WPHY"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in meter per second
                                   'bottom_stress_u': nc.variables["TAUKSI"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)), # in newton per square meter or pascal
                                   'bottom_stress_v': nc.variables["TAUETA"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in newton per square meter or pascal
                                   'bottom_stress_max': nc.variables["TAUMAX"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in newton per square meter or pascal
                                   'eddy_viscosity': nc.variables["VICUV"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in square meter per second
                                   'eddy_diffusivity':nc.variables["DICWW"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in square meter per second
                                   'hydrostatic_pressure':nc.variables["HYDPRES"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in newton per square meter or pascal
                                  'thermocline_depth': nc.variables["THERMOCLINE"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in meter
                                  'water_density':nc.variables["RHO"][time_index, :].data.reshape(np.prod(lon.shape)*len(time_index)),# in kilogram per cubic meter
                                   'velocity_magnitude': velocity_magnitude.reshape(np.prod(lon.shape)*len(time_index))})# in meter per second
    df_coords_untf[df_coords_untf == -999] = pd.NA

    # Add mask according to bathymetry
    #df_mask=pd.concat(list(map(lambda time:(pd.concat([( nc_bathy.z < -15).to_dataframe().reset_index().query('z==False').rename(columns={'x': 'lon', 'y': 'lat'}),pd.DataFrame(dict(zip(df_coords_untf.drop(columns=['time', 'lon', 'lat', 'depth']).columns,[np.nan] * len(df_coords_untf.drop(columns=['time', 'lon', 'lat', 'depth']).columns))),index=[0])], axis=1).assign( time=time, depth=depth))[df_coords_untf.columns],df_coords_untf.dropna(subset=['time']).time.unique()))).reset_index(drop=True)
    #df_mask=pd.concat(list(map(lambda time: (-1 * nc_bathy.z < 10).to_dataframe().reset_index().query('z==True').rename( columns={'x': 'lon', 'y': 'lat'}).assign(water_density=np.nan,thermocline_depth=np.nan,hydrostatic_pressure=np.nan,eddy_diffusivity=np.nan,bottom_stess_max=np.nan,bottom_stress_v=np.nan,bottom_stress_u=np.nan,w=np.nan,temperature=np.nan, u=np.nan, v=np.nan, velocity_magnitude=np.nan, time=time, depth=depth)[df_coords_untf.columns],df_coords_untf.dropna(subset=['time']).time.unique()))).reset_index(drop=True)
    #df_coords_untf = pd.concat([df_coords_untf,df_mask], axis=0).sort_values(['lon','lat','time']).reset_index(drop=True)

    df_gf = gpd.GeoDataFrame(df_coords_untf.dropna(subset=['lat', 'lon']),geometry=gpd.points_from_xy(df_coords_untf.dropna(subset=['lat', 'lon']).lon,df_coords_untf.dropna(subset=['lat', 'lon']).lat), crs=4326)
    # Rasterize Points to Gridded Xarray Dataset

    geo_grid = make_geocube(
        vector_data=df_gf, #[df_gf.time==df_coords_untf.time.unique()[0]]
        measurements=['temperature', 'u', 'v', 'w', 'velocity_magnitude','bottom_stress_u', 'bottom_stress_v', 'bottom_stress_max','eddy_viscosity','eddy_diffusivity', 'hydrostatic_pressure','thermocline_depth', 'water_density'], #
        datetime_measurements=['time'],
        resolution=grid_resolution,  # degrees
        rasterize_function=rasterize_points_griddata,
        group_by="time",
    )

    # Mask raster according to bathymetry
    df_polygon = geo_grid.to_dataframe().reset_index()[['x', 'y']].drop_duplicates(subset=['x', 'y'])
    points = list(map(lambda coords: Point(coords[0], coords[1]), df_polygon.values))
    df_polygon = df_polygon.assign(z=polygon_bathy.contains(points))
    masked_geo_grid = xr.merge([geo_grid, xr.Dataset.from_dataframe(df_polygon.set_index(['x', 'y']))])
    masked_geo_grid = xr.where(masked_geo_grid.z == True, masked_geo_grid, np.nan, keep_attrs=True)
    geo_grid=masked_geo_grid.transpose()

    #fig, ax = plt.subplots()
    #plot=ax.pcolormesh(geo_grid.coords['x'].data,geo_grid.coords['y'].data[::-1],geo_grid.temperature.data[0,::-1],cmap='twilight')
    #fig.colorbar(plot, ax=ax)
    #ax.set_title(pd.to_datetime(nc.variables['time'].values[time_index].tolist())[0])

    x,y=transformer_ws_to_ch.transform(geo_grid.to_dataframe().reset_index().drop_duplicates(['x','y']).x.tolist(), geo_grid.to_dataframe().reset_index().drop_duplicates(['x','y']).y.tolist())
    geo_grid=geo_grid.assign_attrs(swiss_coordinates_x=x,swiss_coordinates_y=y)

    #fig, ax = plt.subplots()
    #plot=ax.pcolormesh(np.array(geo_grid.attrs['swiss_coordinates_x']).reshape(geo_grid.dims['y'],geo_grid.dims['x']),np.array(geo_grid.attrs['swiss_coordinates_y']).reshape(geo_grid.dims['y'],geo_grid.dims['x'])[::-1],geo_grid.thermocline_depth.data[0,::-1],cmap='twilight')
    #fig.colorbar(plot, ax=ax)
    #ax.set_title(pd.to_datetime(nc.variables['time'].values[time_index].tolist())[0])

    geo_grid.variables['temperature'].attrs['units']='degree celsius'
    geo_grid.variables['u'].attrs['units'],geo_grid.variables['u'].attrs['long_name'] ='meter per second', 'horizontal current velocity in the eastward direction'
    geo_grid.variables['v'].attrs['units'],geo_grid.variables['v'].attrs['long_name'] ='meter per second', 'horizontal current velocity in the northward direction'
    geo_grid.variables['w'].attrs['units'],geo_grid.variables['w'].attrs['long_name'] ='meter per second', 'vertical current velocity in the upward direction'
    geo_grid.variables['velocity_magnitude'].attrs['units'],geo_grid.variables['velocity_magnitude'].attrs['long_name'] ='meter per second', 'magnitude of horizontal current velocity'
    geo_grid.variables['bottom_stress_u'].attrs['units'],geo_grid.variables['bottom_stress_u'].attrs['long_name'] ='newton per square meter', 'drag force of horizontal current in the eastward direction'
    geo_grid.variables['bottom_stress_v'].attrs['units'], geo_grid.variables['bottom_stress_v'].attrs['long_name'] = 'newton per square meter', 'drag force of horizontal current in the northward direction'
    geo_grid.variables['bottom_stress_max'].attrs['units'], geo_grid.variables['bottom_stress_max'].attrs['long_name'] = 'newton per square meter', 'maximum drag force of horizontal current'
    geo_grid.variables['eddy_viscosity'].attrs['units'], geo_grid.variables['eddy_viscosity'].attrs['long_name'] = 'square meter per second', 'horizontal eddy viscosity'
    geo_grid.variables['eddy_diffusivity'].attrs['units'], geo_grid.variables['eddy_diffusivity'].attrs['long_name'] = 'square meter per second', 'horizontal eddy diffusivity'
    geo_grid.variables['hydrostatic_pressure'].attrs['units'], geo_grid.variables['hydrostatic_pressure'].attrs['long_name'] = 'newton per square meter', 'Non-hydrostatic pressure'
    geo_grid.variables['thermocline_depth'].attrs['units'], geo_grid.variables['thermocline_depth'].attrs['long_name'] = 'meter', 'depth of the thermocline'
    geo_grid.variables['water_density'].attrs['units'], geo_grid.variables['water_density'].attrs['long_name'] = 'kilogram per cubic meter', 'water density'

    #df_geo_grid=pd.concat([geo_grid.to_dataframe().reset_index(),pd.DataFrame(np.tile(pd.DataFrame(geo_grid.attrs).to_numpy(), (len(time_index), 1)),columns=['swiss_coordinates_x','swiss_coordinates_y','depth'])],axis=1)
    return geo_grid

def download_flow3D_data(year='2024',month='06',path_output=path_to_git / 'data' / 'datafiles' / 'alplakes' / 'flow3d'):
    sundays=pd.date_range(start=str(int(float(year)))+'-'+month.zfill(2), end=str(int(float(year) ))+'-'+str(int(float(month) + 1)).zfill(2), freq='W-SUN').strftime('%Y%m%d').tolist()
    nc_grid=xr.Dataset()
    with requests.Session() as session:
        for sunday in sundays:
            path_to_file =Path(path_output).expanduser() /"delft3d-flow_geneva_{}.nc".format(sunday)
            file_url='https://alplakes-api.eawag.ch/simulations/file/delft3d-flow/geneva/{}'.format(sunday)

            with session.get(file_url, stream=True, headers={'Connection': 'Keep-Alive', 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.181 Safari/537.36'}) as rsp:
                with open(path_to_file, "wb") as f:
                    for a_chunk in rsp.iter_content(chunk_size=131072):  # Loop over content, i.e. eventual HTTP chunks
                        f.write(a_chunk)
            # Open and resterize model ouptput
            nc= xr.open_mfdataset(path_to_file)
            #depth_index = get_closest_index(3.8, np.array(nc.variables["ZK_LYR"][:]) * -1)
            #date_values = nc.variables['time'][:].data
            geo_grid = xr.merge([flow3D_rasterize(nc, time_index=np.arange(nc.dims['time']-1), depth_index=-1,grid_resolution=0.002),flow3D_rasterize(nc, time_index=[-1], depth_index=-1,grid_resolution=0.002)])
            nc.close()
            path_to_file.unlink(missing_ok=True)
            # Mask raster according to bathymetry
            df_polygon=geo_grid.to_dataframe().reset_index()[['x','y']].drop_duplicates(subset=['x','y'])
            points = list(map(lambda coords: Point(coords[0],coords[1]),df_polygon.values))
            df_polygon=df_polygon.assign(z=polygon_bathy.contains(points))
            masked_geo_grid=xr.merge([geo_grid,xr.Dataset.from_dataframe(df_polygon.set_index(['y','x']))])
            masked_geo_grid=xr.where(masked_geo_grid.z==True,masked_geo_grid,np.nan,keep_attrs=True).transpose('time','y','x')
            #fig, ax = plt.subplots()
            #plot=ax.pcolormesh(masked_geo_grid.coords['x'].data,masked_geo_grid.coords['y'].data,masked_geo_grid.temperature.data[0,:],cmap='twilight')
            #fig.colorbar(plot, ax=ax)
            #ax.set_title(masked_geo_grid.variables['time'][-1].values)
            nc_grid=xr.merge([nc_grid,masked_geo_grid],combine_attrs='drop_conflicts')

    nc_grid.to_netcdf('{}/{}.nc'.format(path_to_file.parent,path_to_file.stem[:-2]))

    return nc_grid





# Workflow starts here:

# Sentinel chl-concentration
df_chl=pd.read_csv(path_to_git / 'data' / 'datafiles' / 'alplakes' / 'sentinel' / "geneva_chlorophyll_satellite_summary.csv")
df_chl.columns=['Datetime', 'Satellite', 'Mean_concentration', 'Min_concentration','Max_concentration', 'Pixel_Coverage', 'URL']#['Datetime', 'Satellite', 'Mean (mg m-3)', 'Min (mg m-3)','Max (mg m-3)', 'Pixel Coverage (%)', 'URL']
plot=(ggplot(df_chl.dropna(subset='Max_concentration'))+
      geom_line(mapping=aes(x='Datetime',y='Mean_concentration',group=1))+
      geom_point(mapping=aes(x='Datetime',y='Mean_concentration',group=1),size=0.05)+
      scale_y_continuous(limits=[1,50])+labs(x='',y='Chlorophyll-a concentration (ug/L)')+
      scale_x_datetime(date_breaks='1 months',date_labels='%b',limits=[np.datetime64('2021-03-01'),np.datetime64('2021-11-01')])+
      theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.set_size_inches(6, 4.5)
plot.show()
plot.savefig(fname='{}/figures/alplakes/ts_chl_2021.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


# Simulations 3 D, velocity
import json
json_files = natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'alplakes' / 'flow3d').expanduser().rglob('*.txt')))
df_velocity=pd.DataFrame()
for file in json_files:
    f = open(file, )
    data = json.load(f)
    df_velocity = pd.concat([df_velocity,pd.DataFrame({'datetime': data['time'], 'u': data['variables']['u']['data'], 'v': data['variables']['v']['data']})],axis=0).reset_index(drop=True)
df_velocity['veloctiy_magnitude']=( df_velocity['u'] ** 2 + df_velocity['v'] ** 2 ) ** 0.5
df_velocity['datetime']=pd.to_datetime(df_velocity['datetime'].str[0:-6],format='%Y-%m-%dT%H:%M:%S')
df_velocity['hours']=df_velocity['datetime'].dt.strftime('%H')
plot=(ggplot(df_velocity.query('hours=="09"'))+
      geom_line(mapping=aes(x='datetime',y='veloctiy_magnitude'))+labs(x='',y='Velocity magnitude (m/s)')+theme_paper + theme(text=element_text(size=22))).draw(show=False)
plot.show()
plot.set_size_inches(11.4, 4.5)
plot.savefig(fname='{}/figures/alplakes/velocity_magntiude.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


data_name = natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'alplakes' / 'sentinel').expanduser().rglob('*.tif')))
tiff = rasterio.open(data_name[1])

fig, ax = plt.subplots(figsize = (14, 16))
ax.scatter(6.660835277777778,46.5002275,c='black')
plot=rasterio.plot.show(tiff,cmap='twilight')
im = plot.get_images()[0]
fig.colorbar(im, ax=ax)
plt.show()
plt.savefig(fname='{}/figures/alplakes/map_chl_092021.pdf'.format(str(path_to_git)), dpi=300, bbox_inches='tight')


# ISIMIP model SIMSTRAT, FLOW3D



nc_files=r"D:\Users\dugenne\Downloads\simstrat_gfdl-esm2m_ewembi_picontrol_nosoc_co2_bottemp_geneva_daily_2006_2099.nc4"
nc = netCDF4.Dataset(nc_files)

nc_files=natsorted(list(Path(path_to_git / 'data' / 'datafiles' / 'alplakes' / 'flow3D').expanduser().glob('*.nc')))
nc = xr.open_mfdataset([str(file) for file in nc_files])
lon, lat = transformer.transform(nc.variables["XZ"][:].data,nc.variables["YZ"][:].data)
np.unique(nc.variables["XZ"][:].data)
lon[nc.variables["XZ"][:].data==0]=np.nan
lat[nc.variables["YZ"][:].data==0]=np.nan

plt.figure()
plt.scatter(lon, lat, s=0.01)  # plot grid locations
plt.xlabel("lon")
plt.ylabel("lat")
plt.show()


# Rasterize
depth_index=get_closest_index(3.8, np.array(nc.variables["ZK_LYR"][:]) * -1)
date_values=nc.variables['time'][:].data
geo_grid=flow3D_rasterize(nc,time_index=0,depth_index=82)
fig, ax = plt.subplots()
velo=ax.pcolormesh(geo_grid.coords['x'].data,geo_grid.coords['y'].data[::-1],geo_grid.velocity_magnitude.data[::-1],cmap='twilight')
stream = ax.streamplot(x=geo_grid.coords['x'].data,y=geo_grid.coords['y'].data[::-1],u=geo_grid.variables['u'].data[::-1],v=geo_grid.variables['v'].data[::-1],color='black',density=3,linewidth=1*geo_grid.variables['velocity_magnitude'].data[::-1]/np.nanmax(geo_grid.variables['velocity_magnitude'].data[::-1]))
ax.scatter( 6.660835277777778,46.5002275,s=105,c='red')
fig.show()
def animate(iter):

    geo_grid=flow3D_rasterize(nc,time_index=iter,depth_index=82)
    for artist in ax.collections:
        artist.remove()
    # Clear arrowheads streamplot.
    for artist in ax.get_children():
        if isinstance(artist, FancyArrowPatch):
            artist.remove()

    ax.set_title(geo_grid.attrs['time'])
    velo=ax.pcolormesh(geo_grid.coords['x'].data,geo_grid.coords['y'].data[::-1],geo_grid.velocity_magnitude.data[::-1],cmap='twilight')
    stream = ax.streamplot(x=geo_grid.coords['x'].data,y=geo_grid.coords['y'].data[::-1],u=geo_grid.variables['u'].data[::-1],v=geo_grid.variables['v'].data[::-1],color='black',density=3,linewidth=1*geo_grid.variables['velocity_magnitude'].data[::-1]/np.nanmax(geo_grid.variables['velocity_magnitude'].data[::-1]))
    ax.scatter(6.660835277777778, 46.5002275, s=105, c='red')

    return ax
anim=animation.FuncAnimation(fig, animate, frames=40, blit=False, repeat=False)
anim.save(nc_files[0].parent / 'animation_delft3D-Flow.gif', fps=1)

geo_grid.to_netcdf(nc_files[0].parent /'delftflow3d_geneva_{}_{}.nc'.format(np.datetime_as_string(date_values.min(),unit='D'),np.datetime_as_string(date_values.max(),unit='D')),engine='h5netcdf')
