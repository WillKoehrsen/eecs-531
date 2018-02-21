def apply_blur_filter(blur_filter, image_path):
    
    # Load in the image
    image = Image.open(image_path)
    
    # Crop to correct size
    image = image.crop(box=(0, 0, 
                       int(image.size[0] / blur_filter.shape[0]) * blur_filter.shape[0], 
                       int(image.size[1] / blur_filter.shape[1]) * blur_filter.shape[1]))
    
    im_array = np.array(image)
    
    # Horizontal and vertical moves, using a stride of filter shape
    h_moves = int(im_array.shape[1] / blur_filter.shape[1])
    v_moves = int(im_array.shape[0] / blur_filter.shape[0])
    
    new_image = np.zeros(shape = im_array.shape)
    
    k = np.sum(blur_filter)
    
    # Iterate through 3 color channels
    for i in range(im_array.shape[2]):
        # Extract the layer and create a new layer to fill in 
        layer = im_array[:, :, i]
        new_layer = np.zeros(shape = layer.shape, dtype='uint8')

        # Left and right bounds are determined by columns
        l_border = 0
        r_border = blur_filter.shape[1]


        # Iterate through the number of horizontal and vertical moves
        for h in range(h_moves):
            # Top and bottom bounds are determined by rows
            b_border = 0
            t_border = blur_filter.shape[0]
            for v in range(v_moves):
                patch = layer[b_border:t_border, l_border:r_border]

                # Take the element-wise product of the patch and the filter
                product = np.multiply(patch, blur_filter)

                # Find the weighted average of the patch
                product = np.sum(product) / k
                new_layer[b_border:t_border, l_border:r_border] = product

                b_border = t_border
                t_border = t_border + blur_filter.shape[0]

            l_border = r_border
            r_border = r_border + blur_filter.shape[1]


        new_image[:, :, i] = 255 * ( (new_layer - np.min(new_layer)) / 
                                    (np.max(new_layer) - np.min(new_layer)) )


    # Convert to correct type for plotting
    new_image = new_image.astype('uint8')
    
    plt.imshow(image); plt.title('Original Image'); plt.axis('off')
    plt.show()
    
    plt.imshow(new_image); plt.title('Blurred Image'); plt.axis('off')
    plt.show()
    
    return new_image