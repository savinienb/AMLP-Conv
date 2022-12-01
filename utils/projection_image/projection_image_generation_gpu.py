import math
import tensorflow.compat.v1 as tf



def projection_image_generation_gpu(image, mask, num_angles):

    list=[]
    list_1d=[]



    for ang in range(0,num_angles):
        if ang == 0:

            angle=0


        elif ang != 0:
            angle = angle+(math.pi/num_angles)

        rotate_negative = tf.contrib.image.rotate(
            image,
            -angle,
            interpolation='BILINEAR'
            # name=None
        )

        masked_image = tf.multiply(tf.identity(rotate_negative), mask)
        sum_projection = tf.reduce_sum(masked_image,0)
        mask_sum_projection = tf.reduce_sum(mask,0)
        norm_projection = tf.divide(sum_projection,mask_sum_projection)
        list_1d.append(norm_projection)


        repeat = tf.tile([norm_projection], ([128,1]))
    #repeat2 = tf.tile([repeat], ([128,1]))


        rotate_positive = tf.contrib.image.rotate(
            repeat,
            angle,
            interpolation='BILINEAR'
            # name=None
         )
        # if ang == 0:
        #     list.append(masked_image)

        list.append(rotate_positive)



    return tf.stack(list, axis=0), tf.stack(list_1d, axis=0)

def projection_image_1d(image, mask, num_angles):


    list_1d=[]

    for ang in range(0,num_angles):
        if ang == 0:

            angle=0


        elif ang != 0:
            angle = angle+(math.pi/num_angles)

        rotate_negative = tf.contrib.image.rotate(
            image,
            -angle,
            interpolation='BILINEAR'
            # name=None
        )

        masked_image = tf.multiply(tf.identity(rotate_negative), mask)
        sum_projection = tf.reduce_sum(masked_image,0)
        mask_sum_projection = tf.reduce_sum(mask,0)
        norm_projection = tf.divide(sum_projection,mask_sum_projection)
        list_1d.append(norm_projection)




    return tf.stack(list_1d, axis=0)





def projection_image_generation_gpu_batches(image, mask, num_angles, batch_size):
    mask_tf = tf.convert_to_tensor(mask)[0, ...]
    network_input_2d_stack_list = []
    network_input_1d_stack_list = []
    for i in range(batch_size):
        projection_2d, projection_1d = projection_image_generation_gpu(image[i, 0, ...], mask_tf, num_angles)
        network_input_2d_stack_list.append(projection_2d)
        network_input_1d_stack_list.append(projection_1d)

    return tf.stack(network_input_2d_stack_list, axis=0), tf.stack(network_input_1d_stack_list, axis=0)





def projection_image_1d_batches(image, mask, num_angles, batch_size):
    mask_tf = tf.convert_to_tensor(mask)[0, ...]
    network_input_3d_list = []
    for i in range(batch_size):
        network_input_3d_list.append(projection_image_1d(image[i, 0, ...], mask_tf, num_angles))

    return tf.stack(network_input_3d_list, axis=0)




