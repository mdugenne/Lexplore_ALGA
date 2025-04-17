# Objective: The script aims at processing images from the Cytation

# Modules:
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance
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

# Workflow starts here:
scale_pixel_per_micron=0.852
imagefile_brightfield=list(Path('~/GIT/Lexplore_ALGA/data/datafiles/cytation').expanduser().rglob('bright_*'))

# Grayscale well field
image=ski.io.imread(imagefile_brightfield[0],as_gray=True)
plt.figure(),plt.imshow(image, cmap='gray'),plt.show()
# Multi-channel well field
mc_image=ski.io.imread(str(imagefile_brightfield[0]).replace('bright','Stitched[Tsf_2[DAPI 377,447]+Tsf_2[Chlorophyll A 445,685]+Tsf_2[Bright Field]]'))
plt.figure(),plt.imshow(mc_image),plt.show()

# Step 1 : Discard background pixels located outside the well
thresh = threshold_otsu(image)
bw = closing(image > thresh)
cleared = clear_border(bw)

# label image regions
label_image = label(cleared)
# to make the background transparent, pass the value of `bg_label`,
# and leave `bg_color` as `None` and `kind` as `overlay`
image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
plt.figure(),plt.imshow(image_label_overlay, cmap='gray'),plt.show()

image[(image_label_overlay[:,:,0]!=image_label_overlay[:,:,1])]=0
plt.figure(),plt.imshow(image, cmap='gray'),plt.show()

# Step 2 : Segmentation of the region of interests

edges = ski.feature.canny(image, sigma=0.003)
# plt.figure(),plt.imshow(edges, cmap='gray'),plt.show()
fill_image = sp.ndimage.binary_fill_holes(edges)
fill_image = ski.morphology.erosion(ski.morphology.erosion(ski.morphology.closing(ski.morphology.dilation(ski.morphology.dilation(edges, mode='constant'), mode='constant'))))
fill_image = ski.morphology.remove_small_holes(ski.morphology.closing(fill_image, ski.morphology.square( np.max([1, int(1)]))).astype(bool))

# plt.figure(),plt.imshow(fill_image),plt.show()
label_objects, nb_labels = sp.ndimage.label(fill_image)
labelled_rois_all = measure.label(label_objects)
label_objects= ski.morphology.remove_small_objects(label_objects, min_size=int(np.floor(np.pi*(scale_pixel_per_micron*30/2)**2)))
label_objects, nb_labels = sp.ndimage.label(label_objects)
np.sort(np.bincount(measure.label(label_objects.astype(bool).astype(int)).flat)[1:])
labelled_rois = measure.label(label_objects)
plt.figure(),plt.imshow(labelled_rois),plt.show()
df=pd.DataFrame(ski.measure.regionprops_table(label_image=labelled_rois .astype(int),properties=['slice'])).assign(particle_id=lambda x: x.index+1)

# Step 3 & 4: Determine the number of individual cells within each ROI and count colocated DAPI-positive
df_rois=pd.DataFrame()
for ROI in np.arange(1,nb_labels):

    object_of_interest = labelled_rois == ROI
    df_properties=pd.DataFrame(ski.measure.regionprops_table(label_image=object_of_interest .astype(int), properties=['area', 'area_bbox','extent', 'slice','equivalent_diameter_area'], spacing=scale_pixel_per_micron),index=[ROI])
    ROI_slice=tuple(map(lambda coord:slice(coord.start-50,coord.stop+50),df_properties.loc[ROI,'slice']))
    cropped_image = image.copy()[ROI_slice]

    cropped_image[(labelled_rois[ROI_slice]!=ROI) & (labelled_rois[ROI_slice]!=0)] =np.mean(cropped_image) # Remove pixels outside the ROI
    #plt.figure(), plt.imshow(object_of_interest[ROI_slice]), plt.show()
    plt.figure(), plt.imshow(cropped_image, cmap='gray'), plt.show()
    plt.figure(), plt.imshow(mc_image.copy()[ROI_slice][:,:,1:]), plt.show()

    edges = ski.feature.canny(cropped_image, sigma=0.3)

    plt.figure(),plt.imshow(edges, cmap='gray'),plt.show()

    label_objects, nb_labels = sp.ndimage.label(sp.ndimage.binary_closing(skimage.morphology.dilation(edges ,skimage.morphology.star(4))))
    plt.figure(), plt.imshow(label_objects, cmap='gray'), plt.show()
    labelled_particle=measure.label(label_objects.astype(bool).astype(int))
    largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
    plt.figure(),plt.imshow(largest_object, cmap='gray'),plt.show()
    cropped_mask=np.zeros_like(cropped_image)
    cropped_mask[labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1]=cropped_image[labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1]

    #Check if the ROI is star-shaped (Asterionella)
    av = nfold(largest_object, mod=6, center=tuple(map(lambda val: val.tolist()[0],skimage.measure.regionprops_table(largest_object.astype(bool).astype(int),properties=['centroid']).values())))
    plt.figure(), plt.imshow(av, cmap='gray'), plt.show()
    thresh = skimage.filters.threshold_otsu(av)
    bw =  skimage.morphology.erosion(skimage.morphology.erosion(closing(av > 0.8, skimage.morphology.star((3)))))
    # plt.figure(),plt.imshow(bw, cmap='gray'),plt.show()
    labelled_particle=measure.label(bw.astype(bool).astype(int))
    largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
    #plt.figure(),plt.imshow(largest_object, cmap='gray'),plt.show()

    skeleton_star=ski.morphology.skeletonize(largest_object)
    #plt.figure(),plt.imshow(skeleton_star, cmap='gray'),plt.show()
    skeleton = Skeleton(skeleton_star)
    end_points = skeleton.coordinates[skeleton.degrees == 1]
    star_centroid=tuple(map(lambda val: val.tolist()[0], skimage.measure.regionprops_table(skeleton_star.astype(bool).astype(int),properties=['centroid']).values()))
    if (len(end_points)==6) & (np.std(scipy.spatial.distance.cdist(np.array(star_centroid).reshape((1,2)),end_points))/np.mean(scipy.spatial.distance.cdist(np.array(star_centroid).reshape((1,2)),end_points))<0.1):


        edges = skimage.filters.sobel(cropped_mask)
        # plt.figure(),plt.imshow(edges),plt.show()
        fill_ROI = sp.ndimage.binary_fill_holes(edges)
        # plt.figure(),plt.imshow(fill_ROI),plt.show()

        # find filament end points (where degree of graph nodes is 1)
        skeleton_image=ski.morphology.skeletonize(edges,method='lee' )
        label_objects, nb_labels = sp.ndimage.label(ski.morphology.dilation(ski.morphology.dilation(ski.morphology.dilation(skeleton_image))))
        labelled_particle = measure.label(label_objects.astype(bool).astype(int))
        largest_object = labelled_particle == np.argmax(np.bincount(labelled_particle.flat)[1:]) + 1
        skeleton_image = ski.morphology.skeletonize(largest_object)
        # plt.figure(),plt.imshow(skeleton_image),plt.show()
        skeleton = Skeleton(skeleton_image)
        end_points = skeleton.coordinates[skeleton.degrees == 1]
        plt.figure(), plt.imshow(cropped_image, cmap='gray'),plt.scatter(np.moveaxis(end_points,1,0)[1],np.moveaxis(end_points,1,0)[0],c='red',marker='o'), plt.show()

        #df_rois=pd.concat([df_rois,pd.DataFrame(ski.measure.regionprops_table(label_image=fill_ROI.astype(int),intensity_image=mc_image[ROI_slice],properties=['area','area_convex','area_filled','axis_major_length','axis_minor_length','centroid','equivalent_diameter_area'],spacing=scale_pixel_per_micron)).assign(particle_number=len(end_points),initial_slice=str(ROI_slice),particle_id=ROI)],axis=0).reset_index(drop=True)
        # Count colocated DAPI-positivie particles (chytrids)
        plt.figure(), plt.imshow(mc_image.copy()[ROI_slice][:, :, 1:]), plt.show()
        edges = skimage.filters.sobel(mc_image.copy()[ROI_slice][:, :, 1:][:,:,1])
        #plt.figure(),plt.imshow(edges),plt.show()
        mask_DAPI = skimage.morphology.closing(skimage.morphology.erosion(skimage.morphology.dilation(edges > 0.2)))
        #plt.figure(),plt.imshow(mask_DAPI),plt.show()
        label_DAPI_positive_rois, nb_DAPI_positive_rois = sp.ndimage.label(mask_DAPI)
        plt.figure(),plt.imshow((label_DAPI_positive_rois.astype(bool).astype(int) & fill_ROI)),plt.show()
        label_DAPI_positive_rois, nb_DAPI_positive_rois =sp.ndimage.label((label_DAPI_positive_rois.astype(bool).astype(int) & fill_ROI))
        dapi_particle_slice=tuple(map(lambda x: slice(x[1].start + ski.measure.regionprops_table( label_image=label_DAPI_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0])][0], x[1].start + ski.measure.regionprops_table(label_image=label_DAPI_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0] + 2)][0]),pd.Series(ROI_slice).items()))


        # Count colocated CHL-positivie particles (phytoplankton)
        plt.figure(), plt.imshow(mc_image.copy()[ROI_slice]), plt.show()
        edges_CHL = skimage.filters.sobel(mc_image.copy()[ROI_slice])
        # plt.figure(),plt.imshow(edges_CHL),plt.show()
        mask_CHL = skimage.morphology.closing(skimage.morphology.erosion(skimage.morphology.dilation(edges_CHL[:,:,0] > 0.2)))
        # plt.figure(),plt.imshow(mask_CHL),plt.show()
        labelled_CHL = measure.label(skimage.morphology.dilation(mask_CHL.astype(bool).astype(int)))
        largest_object_CHL = labelled_CHL == np.argmax(np.bincount(labelled_CHL.flat)[1:]) + 1
        # plt.figure(),plt.imshow(largest_object_CHL),plt.show()
        label_CHL_positive_rois, nb_CHL_positive_rois = sp.ndimage.label(largest_object_CHL)
        plt.figure(), plt.imshow(label_CHL_positive_rois), plt.show()
        chl_particle_slice =tuple(map(lambda x: slice(x[1].start + ski.measure.regionprops_table( label_image=label_CHL_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0])][0], x[1].start + ski.measure.regionprops_table(label_image=label_CHL_positive_rois.astype(bool).astype(int), properties=['bbox'])['bbox-{}'.format(x[0] + 2)][0]),pd.Series(ROI_slice).items()))
        plt.figure(), plt.imshow(mc_image.copy()[chl_particle_slice]), plt.show()

        # Filter colocated ROIS only
        #plt.figure(), plt.imshow((label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI)), plt.show()
        #label_CHL_positive_rois, nb_CHL_positive_rois = sp.ndimage.label((label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI))
        # find filament end points (where degree of graph nodes is 1)
        skeleton_image = ski.morphology.skeletonize((label_CHL_positive_rois.astype(bool).astype(int) & fill_ROI))
        # plt.figure(),plt.imshow(skeleton_image),plt.show()
        skeleton = Skeleton(skeleton_image)
        end_points = skeleton.coordinates[skeleton.degrees == 1]
        plt.figure(), plt.imshow(mc_image.copy()[ROI_slice]), plt.scatter(np.moveaxis(end_points, 1, 0)[1],np.moveaxis(end_points, 1, 0)[0], c='white',marker='o'), plt.show()
        df_rois = pd.concat([df_rois, pd.DataFrame(
            ski.measure.regionprops_table(label_image=fill_ROI.astype(int), intensity_image=mc_image[ROI_slice],
                                          properties=['area', 'area_convex', 'area_filled', 'axis_major_length','axis_minor_length', 'centroid', 'equivalent_diameter_area'],
                                          spacing=scale_pixel_per_micron)).assign(chl_particle_number=len(end_points),
                                                                                  dapi_particle_number=nb_DAPI_positive_rois,
                                                                                  initial_slice=str(ROI_slice),
                                                                                  dapi_particle_slice=dapi_particle_slice,
                                                                                  chl_particle_slice=chl_particle_slice,
                                                                                  particle_id=ROI)],axis=0).reset_index(drop=True)
