#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:31:28 2021

@author: kamgoue
"""
import numpy as np 

from PIL import Image
from Dissects.io import (load_NDskl,load_image,load_skeleton,save_skeleton,
                         save_fits,save_image,)

from Dissects.image import (z_project,thinning,dilation)
from Dissects.geometry import Skeleton
from Dissects.draw.plt_draw import (plot_skeleton_3D,plot_face_3D,
                                    plot_junction_3D,)

from Dissects.analysis.analysis import (general_analysis,cellstats)
from Dissects.analysis.analysis_3D_apical import (junction_intensity,
                                                  face_intensity,
                                                  morphology_analysis)
from Dissects.segmentation.seg_3D_apical import Segmentation3D
from Dissects.image.image import (skel_array,greyscale_dilation)

from Dissects.network.network import(network_structure,
                                     create_network,
                                     branch_analysis,
                                     global_network_property,
                                     compute_orientation
                                    )
                                
import os
import time

from skimage import io
import numpy as np
import pandas as pd
import copy
import sys
from skimage import morphology

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

sys.setrecursionlimit(10000)

import plotly.express as px
from scipy.ndimage.morphology import (binary_fill_holes,binary_dilation,)
from scipy import ndimage
import networkx as nx

# update fig parameter
# mpl.rcParams['figure.dpi'],mpl.rcParams['figure.figsize'] = 150,(10,10)

def Skel0(directory,SKELETON_NAME):
    print('Begining Cleaning')
    start = time.time()
    print('----------    -----------------')
    # get original image
    # img0 = load_image(os.path.join(directory,IMAGE_NAME))
    # X_SHAPE,Y_SHAPE,Z_SHAPE  = img0[0].shape[2],img0[0].shape[1],img0[0].shape[0]
    # img_myosin = io.imread(os.path.join(directory, IMAGE_NAME_MYO))    
    cp, fil, point, cp_filament_info, specs = load_NDskl(os.path.join(directory, SKELETON_NAME))
    # #create skeleton object
    skel = Skeleton(cp, fil, point, cp_filament_info, specs)
    print('End Cleaning')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return skel


def Cleaning(directory,SKELETON_NAME):
    print('Begining Cleaning')
    start = time.time()
    print('----------    -----------------')
    # get original image
    # img0 = load_image(os.path.join(directory,IMAGE_NAME))
    # X_SHAPE,Y_SHAPE,Z_SHAPE  = img0[0].shape[2],img0[0].shape[1],img0[0].shape[0]
    # img_myosin = io.imread(os.path.join(directory, IMAGE_NAME_MYO))    
    cp, fil, point, cp_filament_info, specs = load_NDskl(os.path.join(directory, SKELETON_NAME))
    # #create skeleton object
    skel = Skeleton(cp, fil, point, cp_filament_info, specs)
    #Clean skeleton
    skel.remove_lonely_cp()
    skel.remove_free_filament()
    print('End Cleaning')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return skel

def Binary_image(skel):
    print('Begining Binary')
    start = time.time()
    print('----------    -----------------')
    
    # if SAVE_INIT_SEGMENTATION_IMAGE :
    s = ndimage.generate_binary_structure(3,3)
    image_skeleton = skel.create_binary_image()
    image_skeleton = ndimage.morphology.binary_dilation(image_skeleton, structure=s)
    print('End Binary')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return image_skeleton

def Face_Edge_Vert(skel,directory,IMAGE_NAME,X_SIZE,Y_SIZE,Z_SIZE):
    print('Begining Face Edge Vertex')
    start = time.time()
    print('----------    -----------------')
    img0 = load_image(os.path.join(directory,IMAGE_NAME))
    segmentation = Segmentation3D(skel, {"x_size":X_SIZE,
                                         "y_size":Y_SIZE,
                                         "z_size":Z_SIZE,
                                         "x_shape":img0[0].shape[2],
                                         "y_shape":img0[0].shape[1],
                                         "z_shape":img0[0].shape[0]})
    segmentation.generate_segmentation(**segmentation.specs)
    
    print('End Face')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return segmentation

def Images_vert(segmentation,directory,IMAGE_NAME):
    print('Begining Vertex by segmentation')
    start = time.time()
    vertex_image = segmentation.image_identity_vertex(binary=False, dilation_width=3)
    s = ndimage.generate_binary_structure(3,3)
    for i in range(5):
    # # Dilate a little bit vertex to have a better viewing
        vertex_image = ndimage.morphology.binary_dilation(vertex_image, structure=s)
    # save_image(vertex_image, IMAGE_NAME[:-4] + '_output_vertex_test.tif', directory, **segmentation.specs)
    end = time.time()
    print((end-start)/60.0)
    print('End Vertex by segmentation')
    return vertex_image

def Images_junc(segmentation,directory,IMAGE_NAME):
    print('Begining Junct by segmentation')
    start = time.time()
    junction_image = segmentation.image_identity_junction(dilation_width=3, aleatory=True)
    # save_image(junction_image, IMAGE_NAME[:-4] + '_output_junction_test.tif', directory, **segmentation.specs)
    end = time.time()
    print((end-start)/60.0)
    print('End Vertex by segmentation')
    return junction_image

def Images_face(segmentation,directory,IMAGE_NAME,THICKNESS):
    print('Begining face by segmentation')
    start = time.time()
    face_image = segmentation.image_identity_face(aleatory=False, thickness=THICKNESS)
    # save_image(face_image, IMAGE_NAME[:-4] + '_output_face_test.tif', directory, **segmentation.specs)
    end = time.time()
    print((end-start)/60.0)
    print('End Vertex by segmentation')
    return face_image

def Save_csv(directory,vert_df,edge_df,face_df):
    print('Begining Save csv')
    start = time.time()
    print('----------    -----------------')
    vert_df.to_csv(os.path.join(directory, 'vert_df.csv'))
    edge_df.to_csv(os.path.join(directory, 'edge_df.csv'))
    face_df.to_csv(os.path.join(directory, 'face_df.csv'))
    print('End Save csv')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')

def Vertex_image(directory,IMAGE_NAME,vert_df):
    print('Begining Vertex by')
    start = time.time()
    print('----------    -----------------')
    img0 = load_image(os.path.join(directory,IMAGE_NAME))
    vertex_image = np.zeros_like(img0[0])
    s = ndimage.generate_binary_structure(3,3)
    
    vertex_image[vert_df.z_pix.to_numpy().astype(int),
                  vert_df.y_pix.to_numpy().astype(int),
                  vert_df.x_pix.to_numpy().astype(int)] = 1
    # # Dilate a little bit vertex to have a better viewing
    vertex_image = ndimage.morphology.binary_dilation(vertex_image, structure=s)
    s = ndimage.generate_binary_structure(3,3)
    for i in range(5):
    # # Dilate a little bit vertex to have a better viewing
        vertex_image = ndimage.morphology.binary_dilation(vertex_image, structure=s)
    # save_image(vertex_image, IMAGE_NAME[:-4] + '_output_vertex_test.tif', directory, **segmentation.specs)
    end = time.time()
    print('End Vertex')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return vertex_image

def Face_image(directory,myosin_name,segmentation,
               THICKNESS,DILATION,X_SIZE,Y_SIZE,Z_SIZE):
    print('Begining Face')
    start = time.time()
    print('----------    -----------------')
    # #Add junction length to edge_df
    
    # image_skeleton = skel.create_binary_image()
    # image_skeleton = ndimage.morphology.binary_dilation(image_skeleton, structure=s)
    
    from Dissects.analysis.analysis_3D_apical import face_intensity
    img_myosin = load_image(os.path.join(directory,myosin_name))
    enlarge_face = face_intensity(img_myosin[0],segmentation,
                                  THICKNESS, 
                                  DILATION, 
                                  "myosin_intensity")

    enlarge_face_random = np.zeros(enlarge_face.shape)
    unique_value = np.unique(enlarge_face)
    replace_value=[]
    for i in range(len(unique_value)+1):
        rand_val = np.random.randint(10,2**8)
        while rand_val in replace_value:
            rand_val = np.random.randint(10,2**8)
        replace_value.append(rand_val)
    
    for i in range(1, len(unique_value)):
        pos = np.where(enlarge_face==unique_value[i])
        enlarge_face_random[pos] = replace_value[i]
    print('End Face')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return enlarge_face_random

def Tab_analysis(segmentation):
    print('Begining Table')
    start = time.time()
    print('----------    -----------------')
    AREA,PERIMETER,NB_NEIGHBOR,ANISO,J_ORIENTATION = True,True,True,True,True
    
    morphology_analysis(segmentation, area=AREA,
                            perimeter=PERIMETER,nb_neighbor=NB_NEIGHBOR,
                            aniso=ANISO,j_orientation=J_ORIENTATION,)
    
    print('End Table')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return segmentation

def Morpho_analy(segmentation,AREA,PERIMETER,NB_NEIGHBOR,ANISO,J_ORIENTATION):
    print('Begining Morpho_analy')
    start = time.time()
    print('----------    -----------------')
    morphology_analysis(segmentation,area=AREA,perimeter=PERIMETER,nb_neighbor=NB_NEIGHBOR,
                        aniso=ANISO,j_orientation=J_ORIENTATION,)
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return segmentation

def Junct_analy(segmentation,directory,IMAGE_NAME, DILATION):
    print('Begining Junct_analy_intensity')
    start = time.time()
    print('----------    -----------------')
    tiff_junction = io.imread(os.path.join(directory,IMAGE_NAME))
    junction_intensity(tiff_junction, segmentation, DILATION, 'myosin_intensity')
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return segmentation

def Enlarge_fa(segmentation,directory,IMAGE_NAME,THICKNESS, DILATION, label):
    print('Begining Enlarge face intensity')
    start = time.time()
    print('----------    -----------------')
    tiff_junction = io.imread(os.path.join(directory,IMAGE_NAME))
    enlarge_face = face_intensity(tiff_junction, segmentation, THICKNESS, DILATION, label)
    
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return segmentation, enlarge_face

def Apical(segmentation,directory,IMAGE_NAME,THICKNESS, DILATION, label):
    print('Begining Apical intensity')
    start = time.time()
    print('----------    -----------------')
    tiff_junction = io.imread(os.path.join(directory,IMAGE_NAME))
    junction_intensity(tiff_junction, segmentation, DILATION, label)
    enlarge_face = face_intensity(tiff_junction, segmentation, THICKNESS, DILATION, label)
    # output_array = segmentation.image_analyse_junction(label)

    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return enlarge_face

def SaveMorphocsv(segmentation,directory):
    print('Begining SaveMorpho')
    start = time.time()
    print('----------    -----------------')
    segmentation.vert_df.to_csv(os.path.join(directory, 'vert_df_morphology.csv'))
    segmentation.edge_df.to_csv(os.path.join(directory, 'edge_df_morphology.csv'))
    segmentation.face_df.to_csv(os.path.join(directory, 'face_df_morphology.csv'))
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')

def AreaFace(segmentation):
    print('Begining Areaface')
    start = time.time()
    print('----------    -----------------')
    area_face_image = segmentation.image_analyse_face("area")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return area_face_image

def PerimeterFace(segmentation):
    print('Begining Perimeterface')
    start = time.time()
    print('----------    -----------------')
    perim_face_image = segmentation.image_analyse_face("perimeter")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return perim_face_image

def NeighFace(segmentation):
    print('Begining Neighbor')
    start = time.time()
    print('----------    -----------------')
    neig_face_image = segmentation.image_analyse_face("nb_neighbor")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return neig_face_image

def Orientationx(segmentation):
    print('Begining Orientationx')
    start = time.time()
    print('----------    -----------------')
    orientationx_image = segmentation.image_analyse_face("orientationx")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return orientationx_image

def Orientationy(segmentation):
    print('Begining Orientationy')
    start = time.time()
    print('----------    -----------------')
    orientationy_image = segmentation.image_analyse_face("orientationy")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return orientationy_image

def Orientationz(segmentation):
    print('Begining Orientationz')
    start = time.time()
    print('----------    -----------------')
    orientationz_image = segmentation.image_analyse_face("orientationz")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return orientationz_image

def AnisoFace(segmentation):
    print('Begining Anisoface')
    start = time.time()
    print('----------    -----------------')
    aniso_face_image = segmentation.image_aniso()
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return aniso_face_image

def LengthJunc(segmentation):
    print('Begining Lengthjunc')
    start = time.time()
    print('----------    -----------------')
    analyse_junction_image = segmentation.image_analyse_junction("length")
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return analyse_junction_image

def OrientationXY(segmentation):
    print('Begining Orientxy')
    start = time.time()
    print('----------    -----------------')
    segmentation.edge_df['orientation_xy_degree'] = segmentation.edge_df['orientation_xy']*180/np.pi
    segmentation.edge_df['orientation_xy_degree'] = [180+a if a<0 else a for a in segmentation.edge_df['orientation_xy_degree']]
    segmentation.edge_df['orientation_xy_degree'] = [180-a if a>90 else a for a in segmentation.edge_df['orientation_xy_degree']]
    analyse_junction_imagexy = segmentation.image_analyse_junction("orientation_xy_degree", normalize=False, border=False)
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return analyse_junction_imagexy

def OrientationXZ(segmentation):
    print('Begining Orientxz')
    start = time.time()
    print('----------    -----------------')
    segmentation.edge_df['orientation_xz_degree'] = segmentation.edge_df['orientation_xz']*180/np.pi
    segmentation.edge_df['orientation_xz_degree'] = [180+a if a<0 else a for a in segmentation.edge_df['orientation_xz_degree']]
    segmentation.edge_df['orientation_xz_degree'] = [180-a if a>90 else a for a in segmentation.edge_df['orientation_xz_degree']]
    analyse_junction_imagexz = segmentation.image_analyse_junction("orientation_xz_degree", normalize=False, border=False)
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return analyse_junction_imagexz

def OrientationYZ(segmentation):
    print('Begining Orientyz')
    start = time.time()
    print('----------    -----------------')
    segmentation.edge_df['orientation_yz_degree'] = segmentation.edge_df['orientation_yz']*180/np.pi
    segmentation.edge_df['orientation_yz_degree'] = [180+a if a<0 else a for a in segmentation.edge_df['orientation_yz_degree']]
    segmentation.edge_df['orientation_yz_degree'] = [180-a if a>90 else a for a in segmentation.edge_df['orientation_yz_degree']]
    analyse_junction_imageyz = segmentation.image_analyse_junction("orientation_yz_degree", normalize=False, border=False)
    print('End')
    end = time.time()
    print((end-start)/60.0)
    print('----------    -----------------')
    return analyse_junction_imageyz

def Image_myo(directory,IMAGE_NAME_MYO,SKELETON_NAME_MYO_AP):
    # img0_myo = load_image(os.path.join(directory, 
    #                               IMAGE_NAME_MYO))
    
    cp_myo, fil_myo, point_myo, specs_myo, cp_filament_info_myo = load_NDskl(os.path.join(directory, SKELETON_NAME_MYO_AP))
    skel_myo_ap = Skeleton(cp_myo, fil_myo, point_myo, specs_myo, cp_filament_info_myo)
    
    image_myo = skel_array(skel_myo_ap)
    image_myo_ap_dil = greyscale_dilation(image_myo, width=1)
    return image_myo_ap_dil, skel_myo_ap

def Skel_Junc(skel_myo_ap, directory,IMAGE_NAME_MYO, X_SIZE, Y_SIZE, Z_SIZE):
    
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    df_junc_myo = network_structure(skel_myo_ap,  
                               {"X_SIZE": X_SIZE, 
                                "Y_SIZE": Y_SIZE, 
                                "Z_SIZE": Z_SIZE}, 
                               clean=False)
    pixel_size = {"X_SIZE": X_SIZE, "Y_SIZE": Y_SIZE, "Z_SIZE": Z_SIZE}
    df_junc_myo = branch_analysis(df_junc_myo, skel_myo_ap, img_myo, pixel_size)
    df_junc_myo = compute_orientation(df_junc_myo, yz=True)
    df_junc_tosave = df_junc_myo.drop(['points_coords_binaire', 'srce','trgt','length_AU', 'points_coords', 's_xyz','t_xyz'], 
                              axis=1)
    df_junc_tosave.to_csv(os.path.join(directory, 'branch_ntw.csv'))
    dil = 1
    junction_angle = np.zeros(img_myo.shape).T
    for i in range(len(df_junc_myo)):
        junction_angle[tuple((df_junc_myo.points_coords_binaire.loc[i].T).astype(int))]=df_junc_myo.angle_xy.loc[i]
    skel_junction_angle = greyscale_dilation(junction_angle.T, width=dil)
    return skel_junction_angle, df_junc_myo

def Junc_len(directory,IMAGE_NAME_MYO,df_junc_myo):
    dil = 1
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    junction_length = np.zeros(img_myo.shape).T

    for i in range(len(df_junc_myo)):
        junction_length[tuple((df_junc_myo.points_coords_binaire.loc[i].T).astype(int))]=df_junc_myo.length_um.loc[i]

    skel_junction_length = greyscale_dilation(junction_length.T, width=dil)

    return skel_junction_length

def Junc_tortuo(directory,IMAGE_NAME_MYO,df_junc_myo):
    
    from skimage.morphology import dilation
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    dil = 1 #parametre à changer avec un curseur ? 
    
    junction_tortuosity = np.zeros(img_myo.shape).T
    for i in range(len(df_junc_myo)):
        junction_tortuosity[tuple((df_junc_myo.points_coords_binaire.loc[i].T).astype(int))]=df_junc_myo.tortuosity.loc[i]
        
    skel_junction_tortuosity= greyscale_dilation(junction_tortuosity.T,  width=dil)
    
    #test autre dilation
    for i in range(len(df_junc_myo)):
        junction_tortuosity[tuple((df_junc_myo.points_coords_binaire.loc[i].T).astype(int))]=df_junc_myo.tortuosity.loc[i]
    
    footprint = np.array([[[0, 0 ,0], [0, 0, 0], [0, 0 ,0]],
                           [[1, 1 ,1], [1, 1 ,1], [1, 1 ,1]],
                           [[0, 0 ,0], [0, 0 ,0], [0, 0, 0]]
                          ]
                         )
    skel_junction_tortuosity= dilation(junction_tortuosity.T, selem=footprint)
    return skel_junction_tortuosity

def Junc_mean(directory,IMAGE_NAME_MYO,df_junc_myo):
    
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    dil = 1 #parametre à changer avec un curseur ? 
    junction_mean = np.zeros(img_myo.shape).T
    for i in range(len(df_junc_myo)):
        junction_mean[tuple((df_junc_myo.points_coords_binaire.loc[i].T).astype(int))]=df_junc_myo['mean'].loc[i]
    
    skel_junction_mean = greyscale_dilation(junction_mean.T, width=dil)
    return skel_junction_mean

def Centrality(directory,IMAGE_NAME_MYO,df_junc_myo,skel_myo_ap):
    
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    node_df, link_df, G = create_network(df_junc_myo, skel_myo_ap)
    global_ntw_property_df = global_network_property(skel_myo_ap ,df_junc_myo, node_df, img_myo)
    global_ntw_property_df.to_csv(os.path.join(directory, 'global_ntw.csv'))
    node_df['closeness'] = np.nan
    for k, v in nx.closeness_centrality(G).items():
        node_df['closeness'][k]=v
    node_df['betweenness'] = np.nan
    for k, v in nx.betweenness_centrality(G).items():
        node_df['betweenness'][k]=v
    node_df['degree'] = np.nan
    for k, v in nx.degree_centrality(G).items():
        node_df['degree'][k]=v
    
    pd.options.display.float_format = "{:,.10f}".format
    node_df_tosave = node_df.drop(['val','pair','boundary','persistence_ratio','persistence_nsigmas',
                                   'persistence_pair','parent_index','persistence','parent_log_index',
                                   'log_field_value','cell','id_original'], axis=1)

    node_df_tosave.to_csv(os.path.join(directory, 'nodes_ntw.csv'))

    #closeness centrality

    dil = 3 #paramètre modifiable avec un curseur ? 
    centrality= np.zeros(img_myo.shape)
    
    for k, v in nx.degree_centrality(G).items():
        coord_xk0, coord_xk1 = int(node_df.x[k])-dil+1, int(node_df.x[k]) + dil
        coord_yk0, coord_yk1 = int(node_df.y[k])-dil+1, int(node_df.y[k]) + dil
        coord_zk0 = int(node_df.z[k])
        centrality[coord_zk0, coord_yk0:coord_yk1, coord_xk0:coord_xk1] = v
    
    return centrality

def CentralityB(directory,IMAGE_NAME_MYO,df_junc_myo,skel_myo_ap):
    
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    node_df, link_df, G = create_network(df_junc_myo, skel_myo_ap)
    global_ntw_property_df = global_network_property(skel_myo_ap ,df_junc_myo, node_df, img_myo)
    global_ntw_property_df.to_csv(os.path.join(directory, 'global_ntw.csv'))
    node_df['closeness'] = np.nan
    for k, v in nx.closeness_centrality(G).items():
        node_df['closeness'][k]=v
    node_df['betweenness'] = np.nan
    for k, v in nx.betweenness_centrality(G).items():
        node_df['betweenness'][k]=v
    node_df['degree'] = np.nan
    for k, v in nx.degree_centrality(G).items():
        node_df['degree'][k]=v
    
    pd.options.display.float_format = "{:,.10f}".format
    node_df_tosave = node_df.drop(['val','pair','boundary','persistence_ratio','persistence_nsigmas',
                                   'persistence_pair','parent_index','persistence','parent_log_index',
                                   'log_field_value','cell','id_original'], axis=1)

    node_df_tosave.to_csv(os.path.join(directory, 'nodes_ntw.csv'))

    #closeness centrality

    dil = 3 #paramètre modifiable avec un curseur ? 
    centrality= np.zeros(img_myo.shape)
    
    for k, v in nx.betweenness_centrality(G).items():
        coord_xk0, coord_xk1 = int(node_df.x[k])-dil+1, int(node_df.x[k]) + dil
        coord_yk0, coord_yk1 = int(node_df.y[k])-dil+1, int(node_df.y[k]) + dil
        coord_zk0 = int(node_df.z[k])
        centrality[coord_zk0, coord_yk0:coord_yk1, coord_xk0:coord_xk1] = v
    
    return centrality

def CentralityC(directory,IMAGE_NAME_MYO,df_junc_myo,skel_myo_ap):
    
    img0_myo = load_image(os.path.join(directory,IMAGE_NAME_MYO))
    img_myo = img0_myo[0]
    node_df, link_df, G = create_network(df_junc_myo, skel_myo_ap)
    global_ntw_property_df = global_network_property(skel_myo_ap ,df_junc_myo, node_df, img_myo)
    global_ntw_property_df.to_csv(os.path.join(directory, 'global_ntw.csv'))
    node_df['closeness'] = np.nan
    for k, v in nx.closeness_centrality(G).items():
        node_df['closeness'][k]=v
    node_df['betweenness'] = np.nan
    for k, v in nx.betweenness_centrality(G).items():
        node_df['betweenness'][k]=v
    node_df['degree'] = np.nan
    for k, v in nx.degree_centrality(G).items():
        node_df['degree'][k]=v
    
    pd.options.display.float_format = "{:,.10f}".format
    node_df_tosave = node_df.drop(['val','pair','boundary','persistence_ratio','persistence_nsigmas',
                                   'persistence_pair','parent_index','persistence','parent_log_index',
                                   'log_field_value','cell','id_original'], axis=1)

    node_df_tosave.to_csv(os.path.join(directory, 'nodes_ntw.csv'))

    #closeness centrality

    dil = 3 #paramètre modifiable avec un curseur ? 
    centrality= np.zeros(img_myo.shape)
    
    for k, v in nx.closeness_centrality(G).items():
        coord_xk0, coord_xk1 = int(node_df.x[k])-dil+1, int(node_df.x[k]) + dil
        coord_yk0, coord_yk1 = int(node_df.y[k])-dil+1, int(node_df.y[k]) + dil
        coord_zk0 = int(node_df.z[k])
        centrality[coord_zk0, coord_yk0:coord_yk1, coord_xk0:coord_xk1] = v
    
    return centrality
# Analyse croisé
#-----------------------------------
def PrepaCross(directory, SKELETON_NAME_MYO_AP, segmentation, X_SIZE, Y_SIZE, Z_SIZE):
    import pickle
    cp_myo, fil_myo, point_myo, specs_myo, cp_filament_info_myo = load_NDskl(os.path.join(directory, SKELETON_NAME_MYO_AP))
    skel_myo_ap = Skeleton(cp_myo, fil_myo, point_myo, specs_myo, cp_filament_info_myo)

    df_junc_myo = network_structure(skel_myo_ap,  {"X_SIZE": X_SIZE, "Y_SIZE": Y_SIZE, "Z_SIZE": Z_SIZE}, clean=False)
    node_df, link_df, G = create_network(df_junc_myo, skel_myo_ap)
    
    segmentation.update_geom()
    segmentation.compute_normal()
    return segmentation, node_df, G

def max_degree_pc(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k in range(len(node_df)):
        centrality[int(node_df.z.iloc[k]), 
                   int(node_df.y.iloc[k]), 
                   int(node_df.x.iloc[k])] = node_df.nfil.iloc[k]

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross.max()
        
        c.append(np.sum(cross))
        
    face_df['mean_degree']=c
    return sum_centrality

def nb_node2(img, node_df, face_df, it, dil,segmentation):
    
    """
    Create an nd_array. Each dilated apical surface is equal to the number of contained node of signal 2

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um
    pixel_size: dict

    Return
    ------
    nb_nodes : nd_array

    """
    nb_nodes= np.zeros(img.shape)
    nodes = np.zeros(img.shape)
    
    nodes[node_df.z.to_numpy().astype(int),
          node_df.y.to_numpy().astype(int),
          node_df.x.to_numpy().astype(int)] = 1
    nb=[]
    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        #dil_fill_cell_i = ndimage.binary_dilation(fill_cell_i).astype(int)
        cross = nodes*fill_cell_i
        nb_nodes[np.where(fill_cell_i ==1)]= np.count_nonzero(cross)
        nb.append(np.count_nonzero(cross))
                  
    face_df['nb_nodes']=nb
        
    return nb_nodes

def mean_connection2(img, node_df, face_df, it, dil,segmentation):
    """
    Create an nd_array where each dilated apical surface is equal to mean of the number of connection of each node contained in this volume. 
    Fills face_df

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um
    pixel_size: dict

    Return
    ------
    mean_connection : nd_array

    """

    nodes = np.zeros(img.shape)
    for i in range(len(node_df)) :
        if node_df.nfil.iloc[i] != 1:
            nodes[int(node_df.z.iloc[i]), int(node_df.y.iloc[i]), int(node_df.x.iloc[i])] = node_df.nfil.iloc[i]
            
    mean_connection= np.zeros(img.shape)
    nghbr=[]

        
    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = nodes*fill_cell_i

        mean_connection[np.where(fill_cell_i ==1)] = np.sum(cross)/np.count_nonzero(cross)
        nghbr.append(np.sum(cross)/np.count_nonzero(cross))
                  
                  
    face_df['connection_mean'] = nghbr
        
    return mean_connection

def sum_degree_percell2(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.degree_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_degree']=c
    return sum_centrality

def sum_betweenness_percell2(img, node_df, face_df, G, it, dil,segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.betweenness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_btw']=c
    return sum_centrality

def sum_closeness_percell2(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the sum of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.closeness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk= int(node_df.y[k])
        coord_zk= int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= np.sum(cross)*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['sum_closeness']=c
    return sum_centrality

def mean_closeness_pc2(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array, beware: the centrality has been multiplied by the number of node for representation purposes

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.closeness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_closeness']=c
    return sum_centrality

def mean_betweenness_pc2(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.betweenness_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_betweenness']=c
    return sum_centrality

def mean_degree_pc2(img, node_df, face_df, G, it, dil, segmentation):
    """
    Create an nd_array. Each dilated apical surface is equal to the mean of the chosen centrality of signal 2 network contained in this volume

    Parameters
    ----------
    img: nd_array
    node_df: dataframe from create_network
    face_df: segmentation.face_df from dataframe from segmentation
    G : Graph from networkX
    it: integer, iteration of filling 
    dil: float, width of the cell apical surface in um

    Return
    ------
    sum_centrality : nd_array

    """
    
    sum_centrality = np.zeros(img.shape)
    centrality= np.zeros(img.shape)
    c=[]
    
    for k, v in  nx.degree_centrality(G).items():
        coord_xk = int(node_df.x[k])
        coord_yk = int(node_df.y[k])
        coord_zk = int(node_df.z[k])
        centrality[coord_zk, coord_yk, coord_xk] = v

    for i in range(len(face_df)) : 
        cell_i = segmentation.enlarge_face_plane(i, dil)
        fill_cell_i = ndimage.binary_closing(cell_i, iterations = it).astype(int)
        cross = centrality*fill_cell_i
        sum_centrality[np.where(fill_cell_i ==1)]= cross[cross!=0].mean()*len(node_df)
        
        c.append(np.sum(cross))
        
    face_df['mean_degree']=c
    return sum_centrality

def Max_Degre_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    max_degree_image = max_degree_pc(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return max_degree_image

def Nb_Node_Im(directory, IMAGE_NAME, node_df, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    nb_node_pcell = nb_node2(img, node_df, segmentation.face_df, 10, 1, segmentation)
    return nb_node_pcell

def Mean_Connect_Im(directory, IMAGE_NAME, node_df, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    connection = mean_connection2(img, node_df, segmentation.face_df, 10, 1, segmentation)
    return connection

def Sum_Degre_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    sum_degree = sum_degree_percell2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return sum_degree

def Sum_Betweenness_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    sum_between = sum_betweenness_percell2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return sum_between

def Sum_Closenness_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    sum_closen = sum_closeness_percell2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return sum_closen

def Mean_Closenness_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    mean_closen = mean_closeness_pc2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return mean_closen

def Mean_Betweenness_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    mean_between = mean_betweenness_pc2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return mean_between

def Mean_Degre_Im(directory, IMAGE_NAME, node_df, G, segmentation):
    img0 = load_image(os.path.join(directory, IMAGE_NAME))
    img = img0[0]
    mean_degre = mean_degree_pc2(img,node_df,segmentation.face_df, G, 10, 1, segmentation)
    return mean_degre

def plotly_fig2array(fig):
    #convert Plotly fig to  an array
    fig_bytes = fig.to_image(format="png")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)

# directory =  "/home/kamgoue/Rundiss/Seg_test/"


# SKELETON_NAME     = "C1.fits_c1e+03.up.NDskl.BRK.S006.a.NDskl"
# SKELETON_NAME_MYO_AP     = "C2-crop2-GB1.fits_AP_c2e+03.up.NDskl.BRK.S006.a.NDskl"
# IMAGE_NAME        = "C1.tif"
# IMAGE_NAME_MYO = "C2.tif"
# X_SIZE = 0.0458078
# Y_SIZE = 0.0458078
# Z_SIZE = 0.2201818
# tiff_junction = io.imread(os.path.join(directory, 
#                               IMAGE_NAME))
# specs = {"x_size":X_SIZE,
#          "y_size":Y_SIZE,
#          "z_size":Z_SIZE,
#          "x_shape":tiff_junction.shape[2],
#          "y_shape":tiff_junction.shape[1],
#          "z_shape":tiff_junction.shape[0]}
# skel10 = Cleaning(directory,SKELETON_NAME)
# segmentation = Face_Edge_Vert(skel10,directory,IMAGE_NAME,X_SIZE,Y_SIZE,Z_SIZE)
