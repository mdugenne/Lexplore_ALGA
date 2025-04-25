# Objective: The script aims at processing images from the Cytation

# Modules:
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.spatial.distance
from PIL import Image
import skimage
import skimage.color
import skimage.morphology
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from scipy import ndimage as ndi
from skan import Skeleton
from skued import nfold, diffread #conda install scikit-ued
import matplotlib
matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings(action='ignore')

# Workflow starts here:
scale_pixel_per_micron=0.852
imagefile_chlfield=list(Path('~/GIT/Lexplore_ALGA/data/datafiles/cytation').expanduser().rglob('chloro_*'))
df_survey=pd.DataFrame()
for sample in np.arange(len(imagefile_chlfield)):
    # Grayscale well field
    image=skimage.io.imread(imagefile_chlfield[sample],as_gray=True)
    #plt.figure(),plt.imshow(image),plt.show()

    # Step 1 : Discard background pixels located outside the well
    thresh = threshold_otsu(image)
    bw = skimage.morphology.closing(image > thresh)
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
    #plt.figure(),plt.imshow(image_label_overlay, cmap='gray'),plt.show()
    #plt.figure(),plt.imshow((image_label_overlay[:,:,0]!=image_label_overlay[:,:,1]), cmap='gray'),plt.show()

    image[(image_label_overlay[:,:,0]==image_label_overlay[:,:,1])]=0
    image[5500:,:]=0
    #plt.figure(),plt.imshow(image),plt.show()


    # Multi-channel well field
    mc_image=skimage.io.imread(str(imagefile_chlfield[sample]).replace('chloro_','chloroCFW_'))
    mc_shape,image_shape=mc_image.shape,image.shape
    if mc_shape[0:2]>image_shape:
        mc_image = mc_image.copy()[int((tuple(np.array(mc_image.shape[0:2]) - np.array(image.shape[0:2]))[0])/2):-int((tuple(np.array(mc_image.shape[0:2]) - np.array(image.shape[0:2]))[0])/2),int((tuple(np.array(mc_image.shape[0:2]) - np.array(image.shape[0:2]))[1])/2):-int((tuple(np.array(mc_image.shape[0:2]) - np.array(image.shape[0:2]))[1])/2)]
    if mc_shape[0:2]<image_shape:
        image = image.copy()[int((tuple(np.array(image.shape[0:2]) - np.array(mc_image.shape[0:2]))[0])/2):-int((tuple(np.array(image.shape[0:2]) - np.array(mc_image.shape[0:2]))[0])/2),int((tuple(np.array(image.shape[0:2]) - np.array(mc_image.shape[0:2]))[1])/2):-int((tuple(np.array(image.shape[0:2]) - np.array(mc_image.shape[0:2]))[1])/2)]
    #plt.figure(),plt.imshow(mc_image),plt.show()
    #image=mc_image.copy()[:,:,:-1][:,:,0]
    mc_image[(image_label_overlay[:,:,0]==image_label_overlay[:,:,1])]=[0,0,0,255]
    mc_image[5500:, :] = [0,0,0,255]

    # Step 2 : Segmentation of the region of interests

    edges = skimage.feature.canny(image, sigma=0.3)
    # plt.figure(),plt.imshow(edges, cmap='gray'),plt.show()
    fill_image = sp.ndimage.binary_fill_holes(edges)
    fill_image = skimage.morphology.erosion(skimage.morphology.erosion(skimage.morphology.closing(skimage.morphology.dilation(skimage.morphology.dilation(edges, mode='constant'), mode='constant'))))
    fill_image = skimage.morphology.remove_small_holes(skimage.morphology.closing(fill_image, skimage.morphology.square( np.max([1, int(1)]))).astype(bool))

    plt.figure(),plt.imshow(skimage.segmentation.clear_border(fill_image)),plt.show()
    label_objects, nb_labels = sp.ndimage.label(fill_image)
    labelled_rois_all = skimage.measure.label(label_objects)
    label_objects= skimage.morphology.remove_small_objects(label_objects, min_size=int(np.floor(np.pi*(scale_pixel_per_micron*20/2)**2)))
    label_objects, nb_labels_rois = sp.ndimage.label(label_objects)
    np.sort(np.bincount(skimage.measure.label(label_objects.astype(bool).astype(int)).flat)[1:])
    labelled_rois = skimage.measure.label(label_objects)
    plt.figure(),plt.imshow(labelled_rois),plt.show()
    df=pd.DataFrame(skimage.measure.regionprops_table(label_image=labelled_rois .astype(int),properties=['slice'])).assign(particle_id=lambda x: x.index+1)

    # Step 3 & 4: Determine the number of individual cells within each ROI and count colocated DAPI-positive
    df_rois=pd.DataFrame()
    for ROI in np.arange(1,nb_labels_rois):

        object_of_interest = labelled_rois == ROI
        df_properties=pd.DataFrame(skimage.measure.regionprops_table(label_image=object_of_interest .astype(int), properties=['area', 'area_bbox','extent', 'slice','equivalent_diameter_area'], spacing=scale_pixel_per_micron),index=[ROI])
        ROI_slice=tuple(map(lambda coord:slice(np.max([0,coord.start-50]),np.min([coord.stop+50,image.shape[1]])),df_properties.loc[ROI,'slice']))
        cropped_image =image.copy()[ROI_slice]# mc_image.copy()[ROI_slice][:,:,0]

        cropped_image[(labelled_rois[ROI_slice]!=ROI) & (labelled_rois[ROI_slice]!=0)] =np.mean(cropped_image) # Remove pixels outside the ROI
        #plt.figure(), plt.imshow(object_of_interest[ROI_slice]), plt.show()
        #plt.figure(), plt.imshow(cropped_image), plt.show()
        #plt.figure(), plt.imshow(mc_image.copy()[ROI_slice][:,:,1:]), plt.show()#G
        #plt.figure(), plt.imshow(mc_image.copy()[ROI_slice][:,:,0:]), plt.show() #R

        edges = skimage.filters.sobel(cropped_image)#skimage.feature.canny(cropped_image, sigma=0.3)

        #plt.figure(),plt.imshow(edges, cmap='gray'),plt.show()

        label_objects, nb_labels =sp.ndimage.label(sp.ndimage.binary_closing((skimage.morphology.dilation(edges >0.05,skimage.morphology.star(2)))))
        if nb_labels>0:
            #plt.figure(), plt.imshow(label_objects, cmap='gray'), plt.show()
            labelled_particle=skimage.measure.label(label_objects.astype(bool).astype(int))
            largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
            #plt.figure(),plt.imshow(largest_object, cmap='gray'),plt.show()
            cropped_mask=np.zeros_like(cropped_image)
            cropped_mask[labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1]=cropped_image[labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1]

            #Check if the ROI is star-shaped (e.g. Asterionella)
            av = nfold(skimage.morphology.dilation(largest_object,skimage.morphology.star(6)), mod=6, center=tuple(map(lambda val: val.tolist()[0],skimage.measure.regionprops_table(largest_object.astype(bool).astype(int),properties=['centroid']).values())))
            #plt.figure(), plt.imshow(av, cmap='gray'), plt.show()
            thresh = skimage.filters.threshold_otsu(av)
            bw =  sp.ndimage.binary_fill_holes(skimage.segmentation.clear_border(av))#skimage.morphology.erosion(skimage.morphology.erosion(skimage.morphology.closing(av >=0.5, skimage.morphology.star((3)))))
            #plt.figure(),plt.imshow(bw, cmap='gray'),plt.show()
            labelled_particle=skimage.measure.label(skimage.morphology.dilation(bw).astype(bool).astype(int))
            if ((labelled_particle!=0).any()):

                largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
                #plt.figure(),plt.imshow(largest_object, cmap='gray'),plt.show()
                properties=skimage.measure.regionprops_table(largest_object.astype(int),properties=['perimeter','equivalent_diameter_area'])


                if (properties['perimeter'])/(2*np.pi*properties['equivalent_diameter_area']/2)>1.9:
                '''
                skeleton_star=skimage.morphology.skeletonize((largest_object))
                #plt.figure(),plt.imshow(skeleton_star, cmap='gray'),plt.show()
                if len(skeleton_star[skeleton_star==True])>1:
                    skeleton = Skeleton(skeleton_star)
                    end_points = skeleton.coordinates[skeleton.degrees == 1]
                    star_centroid=tuple(map(lambda val: val.tolist()[0], skimage.measure.regionprops_table(skeleton_star.astype(bool).astype(int),properties=['centroid']).values()))
                    if ((len(end_points)/6)/np.floor(len(end_points)/6)==1) & (np.std(scipy.spatial.distance.cdist(np.array(star_centroid).reshape((1,2)),end_points))/np.mean(scipy.spatial.distance.cdist(np.array(star_centroid).reshape((1,2)),end_points))<0.1):
                '''

                        edges = skimage.filters.sobel(cropped_mask)
                        # plt.figure(),plt.imshow(edges),plt.show()
                        fill_ROI = sp.ndimage.binary_fill_holes(skimage.morphology.dilation(edges,skimage.morphology.star(2)))
                        # plt.figure(),plt.imshow(fill_ROI),plt.show()

                        # find filament end points (where degree of graph nodes is 1)
                        skeleton_image=skimage.morphology.skeletonize(edges,method='lee' )
                        label_objects, nb_labels = sp.ndimage.label(skimage.morphology.dilation(skimage.morphology.dilation(skimage.morphology.dilation(skeleton_image))))
                        labelled_particle = skimage.measure.label(label_objects.astype(bool).astype(int))
                        largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
                        skeleton_image = skimage.morphology.skeletonize(largest_object)
                        # plt.figure(),plt.imshow(skeleton_image),plt.show()
                        if len(skeleton_image[skeleton_image == True]) > 1:
                            skeleton = Skeleton(skeleton_image)
                            end_points = skeleton.coordinates[skeleton.degrees == 1]
                            #plt.figure(), plt.imshow(cropped_image, cmap='gray'),plt.scatter(np.moveaxis(end_points,1,0)[1],np.moveaxis(end_points,1,0)[0],c='red',marker='o'), plt.show()
                            #df_rois=pd.concat([df_rois,pd.DataFrame(skimage.measure.regionprops_table(label_image=fill_ROI.astype(int),intensity_image=mc_image[ROI_slice],properties=['area','area_convex','area_filled','axis_major_length','axis_minor_length','centroid','equivalent_diameter_area'],spacing=scale_pixel_per_micron)).assign(particle_number=len(end_points),initial_slice=str(ROI_slice),particle_id=ROI)],axis=0).reset_index(drop=True)

                            # Count colocated DAPI-positivie particles (chytrids)
                            #plt.figure(), plt.imshow(mc_image.copy()[ROI_slice][:, :, 1:]), plt.show()
                            edges = skimage.filters.sobel(mc_image.copy()[ROI_slice][:, :, 1:][:,:,1])
                            #plt.figure(),plt.imshow(edges),plt.show()

                            mask_DAPI = skimage.morphology.remove_small_objects(sp.ndimage.binary_fill_holes(skimage.morphology.erosion(skimage.morphology.dilation(edges > 0.02))), min_size=int( np.floor(np.pi * (scale_pixel_per_micron * 5 / 2) ** 2)))
                            #plt.figure(),plt.imshow(mask_DAPI),plt.show()
                            label_DAPI_positive_rois, nb_DAPI_positive_rois = sp.ndimage.label(mask_DAPI)
                            # Check co-located DAPI-positive ROIs
                            #plt.figure(),plt.imshow((label_DAPI_positive_rois.astype(bool).astype(int) & fill_ROI)),plt.show()
                            label_DAPI_positive_rois, nb_DAPI_positive_rois =sp.ndimage.label((label_DAPI_positive_rois.astype(bool).astype(int) & fill_ROI))
                            properties_dapi=skimage.measure.regionprops_table(label_image=label_DAPI_positive_rois, properties=['bbox'])
                            dapi_particle_slice=[tuple(map(lambda x: slice(x[1].start + properties_dapi['bbox-{}'.format(x[0])][roi_dapi], x[1].start + properties_dapi['bbox-{}'.format(x[0] + 2)][roi_dapi]),pd.Series(ROI_slice).items())) for roi_dapi in np.arange(0,nb_DAPI_positive_rois)] if nb_DAPI_positive_rois>0 else (None,None)


                            # Count colocated CHL-positivie particles (phytoplankton)
                            #plt.figure(), plt.imshow(mc_image.copy()[ROI_slice]), plt.show()
                            edges_CHL = skimage.filters.sobel(mc_image.copy()[ROI_slice])
                            # plt.figure(),plt.imshow(edges_CHL),plt.show()
                            mask_CHL = skimage.morphology.closing(skimage.morphology.erosion(skimage.morphology.dilation(edges_CHL[:,:,0] > 0.05)))
                            # plt.figure(),plt.imshow(mask_CHL),plt.show()
                            labelled_CHL = skimage.measure.label(skimage.morphology.dilation(mask_CHL.astype(bool).astype(int)))
                            if (labelled_CHL ==1).any() :
                                largest_object_CHL = labelled_CHL == np.argmax(np.bincount(labelled_CHL.flat)[1:]) + 1
                                # plt.figure(),plt.imshow(largest_object_CHL),plt.show()
                                label_CHL_positive_rois, nb_CHL_positive_rois = sp.ndimage.label(largest_object_CHL)
                                #plt.figure(), plt.imshow(label_CHL_positive_rois), plt.show()
                                chl_particle_slice =tuple(map(lambda x: slice(x[1].start +skimage.measure.regionprops_table( label_image=label_CHL_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0])][0], x[1].start + skimage.measure.regionprops_table(label_image=label_CHL_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0] + 2)][0]),pd.Series(ROI_slice).items())) if nb_CHL_positive_rois>0 else (None,None)
                                #plt.figure(), plt.imshow(mc_image.copy()[chl_particle_slice]), plt.show()

                                # Filter colocated ROIS only
                                #plt.figure(), plt.imshow((label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI)), plt.show()
                                label_CHL_positive_rois, nb_CHL_positive_rois = sp.ndimage.label((label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI))
                                # find filament end points (where degree of graph nodes is 1)
                                skeleton_image = skimage.morphology.skeletonize(skimage.morphology.dilation(label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI,skimage.morphology.star(2)))
                                # plt.figure(),plt.imshow(skeleton_image),plt.show()
                                if len(skeleton_image[skeleton_image == True]) > 1:
                                    skeleton = Skeleton(skeleton_image)
                                    end_points = skeleton.coordinates[skeleton.degrees == 1]
                                    #plt.figure(),plt.title(ROI), plt.imshow(mc_image.copy()[ROI_slice]), plt.scatter(np.moveaxis(end_points, 1, 0)[1],np.moveaxis(end_points, 1, 0)[0], c='white',marker='o'), plt.show()
                                    df_rois = pd.concat([df_rois, pd.DataFrame(
                                        skimage.measure.regionprops_table(label_image=fill_ROI.astype(int), intensity_image=mc_image[ROI_slice],
                                                                      properties=['area', 'area_convex', 'area_filled', 'axis_major_length','axis_minor_length', 'centroid', 'equivalent_diameter_area'],
                                                                      spacing=scale_pixel_per_micron)).assign(chl_individual_number=len(end_points),
                                                                                                              dapi_individual_number=nb_DAPI_positive_rois,
                                                                                                              initial_slice=str(ROI_slice),
                                                                                                              dapi_particle_slice=str(dapi_particle_slice),
                                                                                                              chl_particle_slice=str(chl_particle_slice),
                                                                                                              particle_id=ROI)],axis=0).reset_index(drop=True)

    # Save ROIs of interest on original stitched image:
    final_image=mc_image.copy()
    dpi=np.prod(Image.open(str(imagefile_chlfield[sample]).replace('chloro_','bright_')).info['dpi'])/10
    height, width, nbands = final_image.shape

    figsize = width / float(dpi), height / float(dpi)

    fig=plt.figure( figsize=figsize), plt.imshow(final_image)

    plt.gca().annotate(text=imagefile_chlfield[sample].parent.stem,xy=(plt.gca().get_xlim()[1],150),xytext=(0, 0),  # 4 points vertical offset.
                            textcoords='offset pixels',
                            ha='right', va='bottom',color='w',size=9*1000/(dpi))
    plt.gca().axis('off')
    for id,rect_slice in df_rois.chl_particle_slice.items():
        if eval(rect_slice)!=(None,None):
            plt.gca().add_patch(plt.Rectangle((eval(rect_slice)[1].start,eval(rect_slice)[0].start),eval(rect_slice)[1].stop-eval(rect_slice)[1].start,eval(rect_slice)[0].stop-eval(rect_slice)[0].start,linewidth=.1,edgecolor='r',facecolor='none'))
            plt.gca().annotate(text=r'ROI: {}, ({},{})'.format(str(df_rois.loc[id,'particle_id']),str(df_rois.loc[id,'chl_individual_number']),str(df_rois.loc[id,'dapi_individual_number'])),
                            xy=( eval(rect_slice)[1].start + (eval(rect_slice)[1].stop-eval(rect_slice)[1].start) / 2,eval(rect_slice)[0].stop),
                            xytext=(0, 0),  # 4 points vertical offset.
                            textcoords='offset pixels',
                            ha='center', va='bottom',color='r',size=1/(dpi*10000))
    for id,rect_slice in df_rois.dapi_particle_slice.items():
        if eval(rect_slice)!=(None,None):
            for roi_dapi in np.arange(0,len(eval(rect_slice))):
                plt.gca().add_patch(plt.Rectangle((eval(rect_slice)[roi_dapi][1].start,eval(rect_slice)[roi_dapi][0].start),eval(rect_slice)[roi_dapi][1].stop-eval(rect_slice)[roi_dapi][1].start,eval(rect_slice)[roi_dapi][0].stop-eval(rect_slice)[roi_dapi][0].start,linewidth=.1,edgecolor='b',facecolor='none'))
    plt.savefig(str(imagefile_chlfield[sample]).replace('chloro','ROI_Stitched[Tsf_2[DAPI 377,447]+Tsf_2[Chlorophyll A 445,685]+Tsf_2[Bright Field]]'), dpi=dpi, transparent=True,  bbox_inches="tight",pad_inches=0)


    # Save dataframe:
    df_sample=df_rois.assign(site=imagefile_chlfield[sample].parent.stem.split('_')[0],sample=imagefile_chlfield[sample].parent.stem,sample_datetime=pd.to_datetime(imagefile_chlfield[sample].parent.stem.split('_')[1],format='%y%m%d'))[['site','sample','sample_datetime']+list(df_rois.columns)]
    df_survey=pd.concat([df_survey,df_sample],axis=0).reset_index(drop=True)