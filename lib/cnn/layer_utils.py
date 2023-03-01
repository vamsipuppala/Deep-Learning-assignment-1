from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class sequential(object):
    def __init__(self, *args):
        """
        Sequential Object to serialize the NN layers
        Please read this code block and understand how it works
        """
        self.params = {}
        self.grads = {}
        self.layers = []
        self.paramName2Indices = {}
        self.layer_names = {}

        # process the parameters layer by layer
        for layer_cnt, layer in enumerate(args):
            for n, v in layer.params.items():
                self.params[n] = v
                self.paramName2Indices[n] = layer_cnt
            for n, v in layer.grads.items():
                self.grads[n] = v
            if layer.name in self.layer_names:
                raise ValueError("Existing name {}!".format(layer.name))
            self.layer_names[layer.name] = True
            self.layers.append(layer)

    def assign(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].params[name] = val

    def assign_grads(self, name, val):
        # load the given values to the layer by name
        layer_cnt = self.paramName2Indices[name]
        self.layers[layer_cnt].grads[name] = val

    def get_params(self, name):
        # return the parameters by name
        return self.params[name]

    def get_grads(self, name):
        # return the gradients by name
        return self.grads[name]

    def gather_params(self):
        """
        Collect the parameters of every submodules
        """
        for layer in self.layers:
            for n, v in layer.params.items():
                self.params[n] = v

    def gather_grads(self):
        """
        Collect the gradients of every submodules
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                self.grads[n] = v

    def apply_l1_regularization(self, lam):
        """
        Gather gradients for L1 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                g1 = (self.params[n]>0).astype(np.float32) - (self.params[n]<0).astype(np.float32)
                self.grads[n] += lam*g1
                ######## END  ########

    def apply_l2_regularization(self, lam):
        """
        Gather gradients for L2 regularization to every submodule
        """
        for layer in self.layers:
            for n, v in layer.grads.items():
                ######## TODO ########
                self.grads[n] += lam*self.params[n]
                ######## END  ########


    def load(self, pretrained):
        """
        Load a pretrained model by names
        """
        for layer in self.layers:
            if not hasattr(layer, "params"):
                continue
            for n, v in layer.params.items():
                if n in pretrained.keys():
                    layer.params[n] = pretrained[n].copy()
                    print ("Loading Params: {} Shape: {}".format(n, layer.params[n].shape))

class ConvLayer2D(object):
    def __init__(self, input_channels, kernel_size, number_filters, 
                stride=1, padding=0, init_scale=.02, name="conv"):
        
        self.name = name
        self.w_name = name + "_w"
        self.b_name = name + "_b"

        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.number_filters = number_filters
        self.stride = stride
        self.padding = padding

        self.params = {}
        self.grads = {}
        self.params[self.w_name] = init_scale * np.random.randn(kernel_size, kernel_size, 
                                                                input_channels, number_filters)
        self.params[self.b_name] = np.zeros(number_filters)
        self.grads[self.w_name] = None
        self.grads[self.b_name] = None
        self.meta = None
    
    def get_output_size(self, input_size):
        '''
        :param input_size - 4-D shape of input image tensor (batch_size, in_height, in_width, in_channels)
        :output a 4-D shape of the output after passing through this layer (batch_size, out_height, out_width, out_channels)
        '''
        output_shape = [None, None, None, None]
        #############################################################################
        # TODO: Implement the calculation to find the output size given the         #
        # parameters of this convolutional layer.                                   #
        #############################################################################
        
        output_shape[0] = input_size[0]
        output_shape[1] = ((input_size[1]+2*self.padding-self.kernel_size)//(self.stride))+1
        output_shape[2] = ((input_size[2]+2*self.padding-self.kernel_size)//(self.stride))+1
        output_shape[3] =  self.number_filters
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return output_shape
#     def conv_mult(self, img_val, filter_weights, b):
# #         ans = np.sum(img_val * filter_weights, axis=(1,2,3))
#         temp = np.multiply(img_val,filter_weights)
#         ans = np.sum(temp)
# #         ans = ans + b
#         return ans
    def do_pad(self, img, pad):
        return np.pad(img, ((0,0),(pad,pad),(pad,pad),(0,0)))

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
#         print(img.shape)
        output_shape = self.get_output_size(img.shape)
        _, input_height, input_width, _ = img.shape
        _, output_height, output_width, _ = output_shape
#         print(self.params['conv_test_w'].shape)

        #############################################################################
        # TODO: Implement the forward pass of a single convolutional layer.       #
        # Store the results in the variable "output" provided above.                #
        #############################################################################
        bt = img.shape[0]
        input_img = self.do_pad(img, self.padding)
        val = np.newaxis
        output = np.zeros((bt,output_height, output_width,  self.number_filters ))
        for i in range(output_height):
            i1 = self.stride * i 
            j1 = i1  + self.kernel_size
            for j in range(output_width):
                i2 = self.stride * j
                j2 = i2 + self.kernel_size
                ans = np.sum(input_img[:,i1:j1, i2:j2,:, val] * self.params[self.w_name][val:,:,:], axis=(1,2,3))
                output[:,i,j,:] = ans
            
        output += self.params[self.b_name]  
        
        
#         for b in range(0, bt):
#             bt_li = []
#             for i in range(output_height):
#                 i1 = self.stride * i 
#                 j1 = i1  + self.kernel_size
#                 for j in range(output_width):
#                     i2 = self.stride * j
#                     j2 = i2 + self.kernel_size
#                     for k in range(self.number_filters):
#                         filter_weights = self.params[self.w_name][:,:,:,k]
#                         bias_values = self.params[self.b_name][k]
#                         one_box = img_pad[b, i1:j1, i2:j2,:]
                        
#                         output[b, i, j, k] = self.conv_mult(one_box, filter_weights, bias_values)
        
            
#         for i in range(output_height):
#             i1 = self.stride * i 
#             j1 = i1  + self.kernel_size
#             for j in range(output_width):
#                 i2 = self.stride * j
#                 j2 = i2 + self.kernel_size
#                 for k in range(self.number_filters):
#                     filter_weights = self.params[self.w_name][:,:,:,k]
#                     bias_values = self.params[self.b_name][k]
#                     one_box = img_pad[:, i1:j1, i2:j2,:]

#                     output[:, i, j, k] = self.conv_mult(one_box, filter_weights, bias_values)
#         output += self.params[self.b_name] 
#         for i in range(output_height):
#             i1 = self.stride * i 
#             j1 = i1  + self.kernel_size
#             for j in range(output_width):
#                 i2 = self.stride * j
#                 j2 = i2 + self.kernel_size
                
#                 filter_weights = self.params[self.w_name][:,:,:,:]
#                 print("***")
#                 print(filter_weights.shape)
#                 bias_values = self.params[self.b_name]
#                 one_box = img[:, i1:j1, i2:j2,:]
#                 print(one_box.shape)
#                 output[:, i, j, :] = self.conv_mult(one_box, filter_weights, bias_values) 
#                 output += self.params[self.b_name] 

                        
                        
                        
                      

            
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img
        
        return output

    def add_pad(self, dimg, img, p):
        te = (0,0)
        dimg_pad = np.pad(dimg, (te,(p,p),(p,p),te))
        img_pad = np.pad(img, (te,(p,p),(p,p),(0,0)))
        return dimg_pad, img_pad
    def backward(self, dprev):
        img = self.meta
        if img is None:
            raise ValueError("No forward function called before for this module!")

        dimg, self.grads[self.w_name], self.grads[self.b_name] = None, None, None
        self.grads[self.b_name] = np.zeros(self.number_filters)
        r1_r2_r3 = (0,1,2)
        self.grads[self.b_name] = np.sum(dprev,axis=r1_r2_r3) 
        val = np.newaxis
        padi = self.padding
        #############################################################################
        # TODO: Implement the backward pass of a single convolutional layer.        #
        # Store the computed gradients wrt weights and biases in self.grads with    #
        # corresponding name.                                                       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        dimg = np.zeros_like(img)
        _, h, w, _ = dprev.shape
        self.grads[self.w_name] = np.zeros((self.kernel_size, self.kernel_size, self.input_channels,self.number_filters))
        
        dimg_p, img_p = self.add_pad(dimg, img, self.padding)
        
        for i in range(h):
            i1 = self.stride * i 
            j1 = i1  + self.kernel_size
            for j in range(w):
                i2 = self.stride * j
                j2 = i2 + self.kernel_size
                self.grads[self.w_name] += np.sum(img_p[:, i1:j1, i2:j2, :, val] *dprev[:, i:i+1, j:j+1, val, :],axis=0)
                dimg_p[:,i1:j1,i2:j2,:] += np.sum(self.params[self.w_name][val,:,:,:,:] * dprev[:,i:i+1,j:j+1,val,:], axis= 4)

        dimg = dimg_p[:,padi:img.shape[1] + padi, padi:img.shape[1] + padi,:]
            
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        self.meta = None
        return dimg


class MaxPoolingLayer(object):
    def __init__(self, pool_size, stride, name):
        self.name = name
        self.pool_size = pool_size
        self.stride = stride
        self.params = {}
        self.grads = {}
        self.meta = None

    def forward(self, img):
        output = None
        assert len(img.shape) == 4, "expected batch of images, but received shape {}".format(img.shape)
        
        #############################################################################
        # TODO: Implement the forward pass of a single maxpooling layer.            #
        # Store your results in the variable "output" provided above.               #
        #############################################################################
        
        _, input_height, input_width, _ = img.shape
        n_H = int(1 + (input_height - self.pool_size) // self.stride)
        n_W = int(1 + (input_width - self.pool_size) // self.stride)
        

        # Initialize output matrix A
        bt = img.shape[0]
        output = np.zeros((bt, n_H,  n_W,  img.shape[3] ))
        for i in range(n_H):
            i1 = self.stride * i 
            j1 = i1  + self.pool_size
            for j in range(n_W):
                i2 = self.stride * j
                j2 = i2 + self.pool_size
                
                output[:, i, j, :] = np.max(img[:,i1:j1,i2:j2,:], axis=(1,2))
        
#         for i in range(n_H):
#             i1 = self.stride * i 
#             j1 = i1  + self.pool_size
#             for j in range(n_W):
#                 i2 = self.stride * j
#                 j2 = i2 + self.pool_size
#                 for k in range(img.shape[3]):
#                     output[:, i, j, k] = np.max(img[:,i1:j1,i2:j2,k])
        
                        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        self.meta = img

        return output
    def fun(self, box, temp):
        return np.multiply((box == np.max(box)), temp)
    

    def backward(self, dprev):
        img = self.meta

        dimg = np.zeros_like(img)
        _, h_out, w_out,_ = dprev.shape
        h_pool, w_pool = self.pool_size,self.pool_size
        
        #############################################################################
        # TODO: Implement the backward pass of a single maxpool layer.              #
        # Store the computed gradients in self.grads with corresponding name.       #
        # Store the output gradients in the variable dimg provided above.           #
        #############################################################################
        for v in range(dprev.shape[0]):
            img_init_values = img[v]
            for i in range(h_out):
                i1 = self.stride * i 
                j1 = i1  + h_pool
                for j in range(w_out):
                    i2 = self.stride * j
                    j2 = i2 + w_pool
                    for k in range(dprev.shape[3]):
                        box = img_init_values[i1:j1,i2:j2,k]
                        dimg[v, i1:j1,i2:j2,k] += self.fun(box, dprev[v, i, j,k])
#                         dimg[v, i1:j1,i2:j2,k] += np.multiply((box == np.max(box)),dprev[v, i, j,k])
       
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dimg
