from PIL import Image
import numpy as np
import os
from scipy.spatial import KDTree
import random
from sklearn.cluster import KMeans
import pandas as pd

def get_avg_pixel(img):
     '''
     Parameters: img (PIL Image object)
     returns: average pixel (RGB tuple)
     '''
     resized_image = img.resize((1,1), Image.Resampling.LANCZOS)
     pixels = resized_image.load()
     return pixels[0,0]

def get_file_avgPixelMap(file_image_map):
    '''
     Parameters:
        file_image_map : (dictionary mapping filenames to their respective PIL Image objects)
     Returns: 
        dictionary mapping filenames to their average calculated pixel value (RGB tuple)
     '''
    file_avgPixel_map = {}
    for file in file_image_map.keys():
        img = file_image_map[file]
        try:
             pixel = get_avg_pixel(img)
             file_avgPixel_map[file] = pixel
        except:
             print("ERROR", file)
    return file_avgPixel_map
        

def get_file_image_map(photos_dir, width=50):
    '''
    Parameters:
        photos_dir : name of directory with photos to make mosaic in it (string)
        width: width in pixels to downscale images to for faster processing and uniform shape (int)
    Returns:
        dictionary mapping filenames to their respective PIL images
    '''
    height = width #square image
    file_image_map = {}
    for filename in os.listdir(photos_dir):
        if filename.endswith(".jpg"):
            img = Image.open(photos_dir+filename)
            img.draft("RGB",(width,height))
            img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            file_image_map[filename] = img
    return file_image_map

def get_concat_h(im1, im2):
    '''
    Parameters: 
        im1: PIL image
        im2: PIL image
    Returns:
        PIL image of im1 and im2 being concatenated together horizontally, im1 on left side of im2
    '''
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    '''
    Parameters: 
        im1: PIL image
        im2: PIL image
    Returns:
        PIL image of im1 and im2 being concatenated together vertically, im1 above im2
    '''
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def get_pixel_files_map(pixel_map):
    '''
    Parameters:
        pixel_map : dictionary mapping filename to average pixel
    Returns:
        dictionary mapping average pixel to a list of filenames with that pixel average
    '''
    pixel_files_map = {}
    #create map from pixel value to list of files with that avg pixel
    for filename in pixel_map.keys():
        avg_pixel_value = pixel_map[filename]
        if avg_pixel_value in pixel_files_map.keys():
            pixel_files_map[avg_pixel_value].append(filename)
        else:
            pixel_files_map[avg_pixel_value] = [filename]
    return pixel_files_map

def get_all_avg_pixels_array(file_image_map):
    '''
    Parameters:
        file_image_map: dictionary mapping filenames to PIL images
    Returns:
        array of all pixel values
    '''
    all_avg_pixels = []
    for filename in file_image_map.keys():
        all_avg_pixels.append(file_image_map[filename])
    return all_avg_pixels

def scale_image_to_basewidth(img, basewidth=50):
    '''
    Parameters:
        img: PIL Image
        basewidth: int
    Returns:
        PIL Image rescaled proportionally such that its new base = basewidth
    '''
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return img

def get_kmeans_dataframe_all_photos(file_image_map):
    '''
    This function performs a kmeans cluster on each image, clustering the array of pixels to identify dominant colors.

    Parameters:
        file_image_map: dictionary mapping filename to respective PIL Image
    Returns:
        Pandas Dataframe linking together:
            filename (string)
            each color cluster centroid (RGB tuple)
            estimated percent of the image falls into that color cluster (float)
    '''

    all_rows = []
    table_columns = ["filename","centroid","percent"]
    for file in file_image_map.keys():
        img = file_image_map[file]

        pixels = np.array(img.getdata())
        kmeans = KMeans(n_clusters=4).fit(pixels)
        centroids = kmeans.cluster_centers_.astype(int)
        
        predictions = kmeans.predict(pixels)
        centroid_indexes, counts = np.unique(predictions, return_counts = True)
        percentages = counts / counts.sum()
        for percent, centroid in zip(percentages, centroids):
            all_rows.append([file, tuple(centroid), percent])
    return pd.DataFrame(all_rows, columns=table_columns)

def string_to_tuple(s):
    '''
    Parameters: 
        s : string in format "(a, b, c)" where a, b, and c are RGB values
    Returns:
        RGB tuple in format (R, G, B)
    '''
    s = s[1:-1]
    myTuple = tuple([int(number) for number in s.split(',')])
    return myTuple

def compare_image_distance(img1, img2):
    '''
    Parameters: 
        img1: PIL Image
        img2: PIL Image
    Returns:
        Mean Squared Error between the pixel values of the two images
    '''
    arr = np.array(img1)
    arr2 = np.array(img2)
    mse = ((arr-arr2)**2).mean().mean()
    return mse

def compare_image_to_images_distances(neighbor_image, window_imgs):
    '''
    Vectorized comparison of an image to an array of images

    Parameters:
        neighbor_image: PIL Image
        window_imgs: array of PIL Images
    Returns:
        array of MSE distances between neighbor_image and each image of window_imgs
    '''
    arr_img = np.array(neighbor_image)
    vector_arr = np.array([np.array(x) for x in window_imgs])
    MSEs = ((vector_arr - arr_img)**2).mean(axis=1).mean(axis=1).mean(axis=1)

    return MSEs

def get_all_image_comparisons_matrix(file_image_map):
    '''
    Generates a matrix of all Mean Squared Errors between every combination of PIL Images.

    Parameters:
        file_image_map : dictionary mapping filename to its respective PIL Image object
    Returns:
        file_index_map : dictionary mapping filename to its index location in the matrix (for quick lookups)
        all_distances : matrix of dimension (num_images X num_images) of all Mean Squared Errors

    '''
    file_index_map = {}
    index = 0
    all_distances = []
    for file in file_image_map.keys():
        file_distances = []
        img = file_image_map[file]
        file_index_map[file] = index
        index += 1
        for compare_file in file_image_map.keys():
            compare_image = file_image_map[compare_file]
            distance = compare_image_distance(img, compare_image)
            file_distances.append(distance)
        all_distances.append(file_distances)
    return file_index_map, all_distances

def make_mosaic_kmeans(target_photo, photos_dir="Photos/", first_run = False):
    '''
    Generates a photo mosaic image of a target photo given a folder of images.
    This method uses kmeans clustering to extract the top 4 dominant colors of an image, and then
    map each pixel of the target image to an image in photos_dir that includes a high yield of a
    dominant color that is similar to that given pixel.

    This approach was experimentally less succesful than the average pixel mapping algorithm, "make_mosaic_avg"

    Parameters:
        target_photo : filename of target photo (string)
        photos_dir : name of folder of images (string)
        first_run : Boolean if this is the first time this method is being run
            This method includes a time-extensive method which should only be run once per photo_dir
    Returns:
        PIL Image of generated photo mosaic
    '''

    #get dictionary mapping each filename to its respective image
    file_image_map = get_file_image_map(photos_dir)
    
    #only run kmeans clustering the first time you do this, so you can use the same set of photos for multiple target images without running this often
    if first_run:
        kmeans_dataframe = get_kmeans_dataframe_all_photos(file_image_map)
        kmeans_dataframe.to_csv("kmean_data_all_photos.csv",index=False)
    else:
        kmeans_dataframe = pd.read_csv("kmean_data_all_photos.csv")
        kmeans_dataframe["centroid"] = kmeans_dataframe["centroid"].map(string_to_tuple)

    #get list of all centroids
    all_centroids = kmeans_dataframe['centroid'].tolist()

    #initialize KD Tree
    nearest_neighbors_tree = KDTree(all_centroids)

    #prep image
    img = Image.open(target_photo)

    img = scale_image_to_basewidth(img, basewidth=50)
    image_matrix = img.load()
    width, height = img.size

    #loop through each pixel of the target image
    for y in range(height):
        for x in range(width):
            pixel = image_matrix[x,y]
            #find the 10 photos that have a dominant centroid most near the current pixel
            nearest_neighbors = nearest_neighbors_tree.query(pixel, k = 10)
            
            #generate subset of kmeans_dataframe that includes only the top-10 nearest neighbors
            closest_files_dfs = []
            for i in range(10):    
                pixel_index = nearest_neighbors[1][i]#[random.randint(0, 9)]
                neighbor_pixel = all_centroids[pixel_index]
                closest_files = kmeans_dataframe[kmeans_dataframe['centroid'] == neighbor_pixel]#["filename"].values
                closest_files_dfs.append(closest_files)
            closest_files_df = pd.concat(closest_files_dfs)

            # pick a random image of the top 10 centroid matches
            closest_file = closest_files_df.sort_values("percent",ascending=False)['filename'].values[random.randint(0, 9)]
            
            #gets the image given the filename
            closest_img = file_image_map[closest_file]

            #add new image object running row image
            if x == 0:
                row_image = closest_img
            else:
                row_image = get_concat_h(row_image, closest_img)
        #add row image object to running column image
        if y == 0:
            column_image = row_image
        else:
            column_image = get_concat_v(column_image, row_image)
    #return stack of images as one
    return column_image

def make_mosaic_avg(target_photo, photos_dir="Photos/", basewidth=50, pixelWidth=50):
    '''
    This function takes in a target_photo filename, a photos_dir directory name, the basewidth of pixels for the target image to be reshaped to,
    and the pixelWidth, which is the width of each image replacing each pixel in the resized target image
    '''
    #maps filenames to PIL image objects
    print("generating compressed file image map...")
    file_image_map = get_file_image_map(photos_dir, width = 50)
    print("generating high-res file image map...")
    file_image_map_hifi = get_file_image_map(photos_dir, width=pixelWidth)

    #maps filenames to the file's average pixel value
    print("generating file avgPixel map...")
    file_avgPixel_map = get_file_avgPixelMap(file_image_map)

    #maps avg pixel values back to filenames
    print("getting pixel files map...")
    pixel_files_map = get_pixel_files_map(file_avgPixel_map)

    #array of all average pixels. There should be one element per file.
    print('getting all avg pixels array...')
    all_avg_pixels = get_all_avg_pixels_array(file_avgPixel_map)

    #gets a matrix of all image similarity comparisons, as well as a mapping from filename to index in the matrix. 
    print('computing all MSEs...')
    file_index_map, all_distances_matrix = get_all_image_comparisons_matrix(file_image_map)

    #initialize datatype for quick lookup of similar pixels
    print('initializing KDtree...')
    nearest_neighbors_tree = KDTree(all_avg_pixels)

    #load and prep image
    img = Image.open(target_photo)
    img = scale_image_to_basewidth(img, basewidth=basewidth)
    image_matrix = img.load()
    width, height = img.size
    
    #initialize matrix to be used to check nearby appended images so as to not add a duplicate
    filename_matrix = []

    #basically window size in sliding window algorithm
    min_array_distance_between_similar_photos = 5
    #loop through each pixel of the target image
    print("generating mosaic...")
    for y in range(height):
        filename_matrix.append([])
        for x in range(width):
            pixel = image_matrix[x,y]
            #select 200 nearest neighbors
            nearest_neighbors_indexes = nearest_neighbors_tree.query(pixel, k = 200)[1]
            #loop through sliding window of filenames appended nearby to ensure no similar images are near
            valid_image_found = False
            for pixel_index in nearest_neighbors_indexes:
                if not valid_image_found:
                    #search for a neighbor that fits the constraint of not being in the last min_distance rows or columns, then break
                    neighbor_pixel = all_avg_pixels[pixel_index]
                    neighbor_file = pixel_files_map[neighbor_pixel][0]
                    neighbor_image = file_image_map[neighbor_file]
                    neighbor_distance_index = file_index_map[neighbor_file]

                    window_x_min = max(0, x-min_array_distance_between_similar_photos)
                    window_y_min = max(0, y-min_array_distance_between_similar_photos)
                    window_x_max = min(width-1, x+min_array_distance_between_similar_photos)
                    if x == 0 and y == 0:
                       image_toAdd = neighbor_image
                       filename_toAdd = neighbor_file
                    else: 
                        image_toAdd = 4 #this is for debugging to ensure we don't accidentally recycle this variable in the loop
                        isValid = True #starts true, then gets disqualified potentially
                        window_imgs = []
                        for window_y in range(window_y_min, y+1):
                            if not valid_image_found:
                                for window_x in range(window_x_min, window_x_max + 1):
                                    if not valid_image_found:
                                        if not (window_y == y and window_x >= x):
                                            window_file = filename_matrix[window_y][window_x]
                                            window_file_index = file_index_map[window_file]
                                            image_distance = all_distances_matrix[neighbor_distance_index][window_file_index]
                                            #arbitrary threshold that has good experimental results
                                            if image_distance < 85:
                                                isValid = False
                        #if isValid is still true, then set flags to continue out of these loops
                        if isValid:
                            valid_image_found = True
                            image_toAdd = file_image_map_hifi[neighbor_file]# neighbor_image
                            filename_toAdd = neighbor_file
            #add image_toAdd to row
            if x == 0:
                row_image = image_toAdd
            else:
                row_image = get_concat_h(row_image, image_toAdd)
            
            #add filename of appended image to filename_matrix
            filename_matrix[y].append(filename_toAdd)
        
        #append row of images to an image of all previous columns
        if y == 0:
            column_image = row_image
        else:
            column_image = get_concat_v(column_image, row_image)
    #return all columns of images of rows of images appended together
    return column_image

def optimize_pixelWidth_for_printer(printer_dpi = 12000, printer_inches = 24, basewidth = 60):
    '''
    Return the maximum width resolution that the mosaic images can be without losing data in printer
    note: large pixelWidths may consume a massive amount of RAM and cause crashes

    Parameters:
        printer_dpi : printer's horizontal dots per inch (int)
        printer_inches : printer's horizontal page size (int)
        base_width : width that target image's base will be scaled down to in pixels (int)
    Returns: 
        pixelWidth : width in pixels that the mosaic photos should be scaled to before being replacing target image pixels
    '''
    printer_pixels = printer_dpi * printer_inches
    pixels_per_row = basewidth
    pixelWidth = printer_pixels // pixels_per_row
    return pixelWidth

pixelWidth = 50
basewidth = 50
target_photo = "husky_flowers.jpg"
photos_dir = "Dog_Photos/" 

img = make_mosaic_avg(target_photo, photos_dir, basewidth=basewidth, pixelWidth=pixelWidth)
img.save("mosaic.jpg")
