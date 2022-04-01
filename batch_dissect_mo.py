import numpy as np
from tifffile import (imsave, imread)
import os
from basemodule import (Cleaning, Images_vert, Vertex_image, Images_junc)
from basemodule import (Binary_image, Images_face, Face_Edge_Vert, Face_image)
from basemodule import (Skel_Junc, Junc_len, Image_myo, Apical)
from basemodule import (Junc_tortuo, Junc_mean, Centrality, CentralityB, CentralityC)
from basemodule import (AreaFace, PerimeterFace, AnisoFace, NeighFace, Morpho_analy)
from basemodule import (OrientationXY, OrientationYZ, OrientationXZ, Morpho_analy, LengthJunc)
from basemodule import (PrepaCross, Max_Degre_Im, Nb_Node_Im, Mean_Connect_Im, Sum_Degre_Im, Mean_Degre_Im)
from basemodule import (Sum_Betweenness_Im, Sum_Closenness_Im, Mean_Closenness_Im, Mean_Betweenness_Im)
import pickle
import time



#------------------------------- Segmentaion-------------------
def DoSeg(fileskel,originalfil,voxelX0,voxelY0,voxelZ0,thick0):
    print("I begin segmentation")
    start = time.time()
    workingdir = os.path.dirname(fileskel)
    file = os.path.basename(fileskel)
    cud, newd = os.getcwd(), file[:-5]
    os.chdir(workingdir)
    if os.path.isdir('Arrays_' + newd):
        os.chdir('Arrays_' + newd)
        print('Segmentation analysis already done')
        bb1 = np.load('skelvert.npz')
        skelbin = bb1['binary']
        skelbin = np.array(skelbin).astype(float)
        bb2 = bb1['vertx']
        vertx = 5*np.array(bb2).astype(float)
        skelvert = skelbin + vertx
    
        bb03 = bb1['facej']
        facej = np.array(bb03).astype(float)
    
        bb3 = bb1['face_image']
        face_image = np.array(bb3).astype(float)
    else:
        skelet = Cleaning(workingdir,fileskel)
        segmentation = Face_Edge_Vert(skelet,workingdir,originalfil,float(voxelX0), float(voxelY0),float(voxelZ0))
                
        binary = Binary_image(skelet)
        face_image = Images_face(segmentation, workingdir, originalfil, float(thick0)) 
        vertx = Vertex_image(workingdir,originalfil,segmentation.vert_df)
        facej = Images_junc(segmentation,workingdir,originalfil)
        os.mkdir('Arrays_' + newd)
        os.chdir('Arrays_' + newd)
        np.savez_compressed('skelvert.npz', binary=binary, face_image=face_image, vertx=vertx, facej=facej)
        with open('segmentation.pkl', 'wb') as outp:
            pickle.dump(segmentation, outp, pickle.HIGHEST_PROTOCOL)
        with open('segmentation4cross.pkl', 'wb') as outp:
                pickle.dump(segmentation, outp, pickle.HIGHEST_PROTOCOL)

        skelvert = binary.astype(float) + 5*(vertx.astype(float))
        facej = facej.astype(float)
        face_image = face_image.astype(float)
    os.chdir(cud)
    end = time.time()
    tt = (end-start)/60.0
    print("End Segmentation: "+str(tt))
    return skelvert, facej, face_image
#-------------------------Reseaux-------------------------------------
def DoNet(fileskel2, originalfil2, voxelX2, voxelY2, voxelZ2):
    print("I begin Begin Network")
    start = time.time()
    workingdir2 = os.path.dirname(fileskel2)
    file2 = os.path.basename(fileskel2)
    cud, newd2 = os.getcwd(), file2[:-5]
    os.chdir(workingdir2)
    if os.path.isdir('Arrays_' + newd2):
        os.chdir('Arrays_' + newd2)
        print('Network analysis already done')
        bb1 = np.load('netw.npz')
        skel = np.array(bb1['skelmyo']).astype(float)
        jorient = np.array(bb1['jorient']).astype(float)
        jlength = np.array(bb1['jlength']).astype(float)
        jtortuo = np.array(bb1['jtortuo']).astype(float)
        jmean = np.array(bb1['jmean']).astype(float)
        jcentr = np.array(bb1['jcentr']).astype(float)
        jcentrb = np.array(bb1['jcentrb']).astype(float)
        jcentrc = np.array(bb1['jcentrc']).astype(float)
        mmaxd, mmaxb, mmaxc = jcentr.max(), jcentrb.max(), jcentrc.max()
        jcentr = jcentr + (mmaxd + 1)*skel
        jcentrb = jcentrb + (mmaxb + 1)*skel
        jcentrc = jcentrc + (mmaxc + 1)*skel
        im2 = imread(originalfil2)
        im2 = im2.astype(float)
    else:
        image_myo_ap_dil, skel_myo_ap = Image_myo(workingdir2, originalfil2, fileskel2)
        binarymyo = Binary_image(skel_myo_ap); skel = np.array(binarymyo).astype(float)
        jorient, df_junc_myo = Skel_Junc(skel_myo_ap, workingdir2,originalfil2, float(voxelX2), float(voxelY2), float(voxelZ2))
        jorient = np.array(jorient).astype(float)
        jlength = Junc_len(workingdir2,originalfil2,df_junc_myo)
        jlength = np.array(jlength).astype(float)
        jtortuo = Junc_tortuo(workingdir2,originalfil2,df_junc_myo)
        jtortuo = np.array(jtortuo).astype(float)
        jmean = Junc_mean(workingdir2, originalfil2, df_junc_myo)
        jmean = np.array(jmean).astype(float)
        jcentr = Centrality(workingdir2,originalfil2,df_junc_myo,skel_myo_ap)
        jcentr = np.array(jcentr).astype(float)
        jcentrb = CentralityB(workingdir2,originalfil2,df_junc_myo,skel_myo_ap)
        jcentrb = np.array(jcentrb).astype(float)
        jcentrc = CentralityC(workingdir2,originalfil2,df_junc_myo,skel_myo_ap)
        jcentrc = np.array(jcentrc).astype(float)
        im2 = imread(originalfil2)
        im2 = im2.astype(float)
        os.mkdir('Arrays_' + newd2)
        os.chdir('Arrays_' + newd2)
        np.savez_compressed('netw.npz', jorient=jorient, jlength=jlength,jtortuo=jtortuo, jmean=jmean, 
        jcentr = jcentr, jcentrb = jcentrb, jcentrc = jcentrc, skelmyo = binarymyo)
        
    os.chdir(cud)
    end = time.time()
    tt = (end-start)/60.0
    print("End Network: "+str(tt))
    return im2, skel, jorient, jlength, jtortuo, jmean, jcentr, jcentrb, jcentrc
#----------------------------Morphologie--------------------------------------------------
def DoMorph(fileskel):
    print("I begin Morphology")
    start = time.time()
    workingdir = os.path.dirname(fileskel)
    file = os.path.basename(fileskel)
    cud, newd = os.getcwd(), file[:-5]
    os.chdir(workingdir)
    os.chdir('Arrays_' + newd)
    if os.path.isfile('morpho.npz'):
        print('Morpho analysis already done')
        bb1 = np.load('morpho.npz')
        area = bb1['area']
        perimeter = bb1['perimeter']
        neig = bb1['neig']
        aniso = bb1['aniso']
    else:
        with open('segmentation.pkl', 'rb') as inp:
            segmentation = pickle.load(inp)
        rbtar, rbtpr, rbtnb, rbtan, rbtjo = True, True, True, True, True
        segmentation = Morpho_analy(segmentation,rbtar,rbtpr,rbtnb,rbtan,rbtjo)
        with open('segmentation.pkl', 'wb') as outp:
            pickle.dump(segmentation, outp, pickle.HIGHEST_PROTOCOL)
        with open('segmentation2.pkl', 'wb') as outp:
            pickle.dump(segmentation, outp, pickle.HIGHEST_PROTOCOL)
        area = AreaFace(segmentation)
        perimeter = PerimeterFace(segmentation)
        neig = NeighFace(segmentation)
        aniso = AnisoFace(segmentation)
        np.savez_compressed('morpho.npz', area=area, perimeter=perimeter, neig=neig, aniso=aniso)
    os.chdir(cud)
    end = time.time()
    tt = (end-start)/60.0
    print("End Morphology: "+str(tt))
    return area, perimeter, neig, aniso
#----------------------------Junction analyse--------------------------------------------------
def DoJun(fileskel):
    print("I begin junction analysis")
    start = time.time()
    workingdir = os.path.dirname(fileskel)
    os.chdir(workingdir)
    file = os.path.basename(fileskel)
    cud, newd = os.getcwd(), file[:-5]
    os.chdir('Arrays_' + newd)
    if os.path.isfile('junc.npz'):
        print('Junction analysis already done')
        bb1 = np.load('junc.npz')
        length = bb1['length']
        orientxz = bb1['orientxz']
        orientyz = bb1['orientyz']
    else:
        with open('segmentation.pkl', 'rb') as inp:
            segmentation = pickle.load(inp)
        length = LengthJunc(segmentation)
        orientxz = OrientationXZ(segmentation)
        orientyz = OrientationYZ(segmentation)
        np.savez_compressed('junc.npz', orientxz=orientxz, orientyz=orientyz, length=length)
    os.chdir(cud)
    end = time.time()
    tt = (end-start)/60.0
    print("End junction: "+str(tt))
    return length, orientxz, orientyz
#----------------------------Analyse Croisée--------------------------------------------------
def DoCros(fileskel, fileskel2):
    print("I begin cross analysis")
    start = time.time()
    workingdir2 = os.path.dirname(fileskel2)
    workingdir = os.path.dirname(fileskel)
    file = os.path.basename(fileskel)
    file2 = os.path.basename(fileskel2)
    cud, newd, newd2 = os.getcwd(), file[:-5], file2[:-5]
    os.chdir(workingdir2)
    os.chdir('Arrays_' + newd2)
    if os.path.isfile('cross.npz'):
        print('Cross analysis already done')
        bb1 = np.load('cross.npz')
        max_degree_image = np.array(bb1['max_degree_image']).astype(float)
        nb_node_pcell = np.array(bb1['nb_node_pcell']).astype(float)
        connection = np.array(bb1['connection']).astype(float)
        sum_degree = np.array(bb1['sum_degree']).astype(float)
        sum_between = np.array(bb1['sum_between']).astype(float)
        sum_closen = np.array(bb1['sum_closen']).astype(float)
        mean_closen = np.array(bb1['mean_closen']).astype(float)
        mean_between = np.array(bb1['mean_between']).astype(float)
        mean_degre = np.array(bb1['mean_degre']).astype(float)
    else:
        os.chdir(workingdir)
        os.chdir('Arrays_' + newd)
        with open('segmentation4cross.pkl', 'rb') as inp:
            segmentation = pickle.load(inp)
        os.chdir(cud)
        os.chdir(workingdir2)
        os.chdir('Arrays_' + newd2)

        segmentation, node_df, G = PrepaCross(workingdir2, fileskel2, segmentation, float(voxelX2), float(voxelY2), float(voxelZ2))
        max_degree_image = Max_Degre_Im(workingdir, originalfil, node_df, G, segmentation)
        nb_node_pcell = Nb_Node_Im(workingdir, originalfil, node_df, segmentation)
        connection = Mean_Connect_Im(workingdir, originalfil, node_df, segmentation)
        sum_degree = Sum_Degre_Im(workingdir, originalfil, node_df, G, segmentation)
        sum_between = Sum_Betweenness_Im(workingdir, originalfil, node_df, G, segmentation)
        sum_closen = Sum_Closenness_Im(workingdir, originalfil, node_df, G, segmentation)
        mean_closen = Mean_Closenness_Im(workingdir, originalfil, node_df, G, segmentation)
        mean_between = Mean_Betweenness_Im(workingdir, originalfil, node_df, G, segmentation)
        mean_degre = Mean_Degre_Im(workingdir, originalfil, node_df, G, segmentation)
        np.savez_compressed('cross.npz', max_degree_image=max_degree_image, nb_node_pcell=nb_node_pcell,
                            connection=connection, sum_degree=sum_degree, sum_between=sum_between, sum_closen=sum_closen,
                            mean_closen=mean_closen, mean_between=mean_between, mean_degre=mean_degre)

    os.chdir(cud)
    end = time.time()
    tt = (end-start)/60.0
    print("End Cross Analysis: "+str(tt))
    return max_degree_image, nb_node_pcell, connection, sum_degree, sum_between, sum_closen, mean_closen, mean_between, mean_degre

def DoApi(fileskel1,directory,IMAGE_NAME,THICKNESS,DILATION, label):
    print("I begin Apical analysis")
    start = time.time()
    file = os.path.basename(fileskel1)
    cud, newd = os.getcwd(), file[:-5]
    workingdir = os.path.dirname(fileskel1)
    os.chdir(workingdir)
    os.chdir('Arrays_' + newd)
    if os.path.isfile(label+'.npz'):
        bb1 = np.load(label+'.npz')
        mlabel = np.array(bb1['label']).astype(float)
    else:
        with open('segmentation4cross.pkl', 'rb') as inp:
            segmentation = pickle.load(inp)
        enlarge_face = Apical(segmentation,directory,IMAGE_NAME,THICKNESS, DILATION, label)
        mlabel = np.array(enlarge_face).astype(float)
        np.savez_compressed(label+'.npz', label=mlabel)
    os.chdir(cud)
    
    end = time.time()
    tt = (end-start)/60.0
    print("End Apical: "+str(tt))
    return mlabel

####################
## Entrez les parametres Segmentation
#############################################
## Repertoire principal
workingdir = "/home/kamgoue/Rundiss/Seg_test"
## Fichier squelette
fileskel = "C1.fits_c1e+03.up.NDskl.BRK.S006.a.NDskl"
fileskel = os.path.join(workingdir,fileskel)
## Fichier original
originalfil = "C1.tif"
originalfil = os.path.join(workingdir,originalfil)
## Voxelisation
voxelX0, voxelY0, voxelZ0 = 0.0458078, 0.0458078, 0.2201818
## Thickness
thick0 = 0.5
## Junction dilatation
juncdil0 = 3

####################
## Entrez les parametres Reseaux
#############################################
## Repertoire principal
workingdir2 = "/home/kamgoue/Rundiss/Networks"
## Fichier squelette
fileskel2 = "C2.fits_AP_c2e+03.up.NDskl.BRK.S006.a.NDskl"
fileskel2 = os.path.join(workingdir2,fileskel2)
## Fichier original
originalfil2 = "C2.tif"
originalfil2 = os.path.join(workingdir2,originalfil2)
## Voxelisation reseaux
voxelX2, voxelY2, voxelZ2 = 0.0458078, 0.0458078, 0.2201818

skelvert0, facej0, face_image0 = DoSeg(fileskel,originalfil,voxelX0,voxelY0,voxelZ0,thick0)
im2, skel, jorient, jlength, jtortuo, jmean, jcentr, jcentrb, jcentrc = DoNet(fileskel2, originalfil2, voxelX2, voxelY2, voxelZ2)
area, perimeter, neig, aniso = DoMorph(fileskel)
# length, orientxz, orientyz = DoJun(fileskel)
enlarge_face = DoApi(fileskel,workingdir2,"C2.tif",thick0,juncdil0, 'labeltest')