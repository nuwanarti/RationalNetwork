import tensorflow as tf
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers, regularizers, constraints


class CustomDense(Layer):

    def __init__(self, alpha_initializer=[1.1915, 1.5957, 0.5, 0.0218], beta_initializer=[2.383, 0.0, 1.0], 
                 alpha_regularizer=None, beta_regularizer=None, alpha_constraint=None, beta_constraint=None,
                 shared_axes=None, **kwargs):
        super(CustomDense, self).__init__(**kwargs)
        self.supports_masking = True

        # Degree of rationals
        self.degreeP = len(alpha_initializer) - 1
        self.degreeQ = len(beta_initializer) - 1
        
        # Initializers for P
        self.alpha_initializer = [initializers.Constant(value=alpha_initializer[i]) for i in range(len(alpha_initializer))]
        self.alpha_regularizer = regularizers.get(alpha_regularizer)
        self.alpha_constraint = constraints.get(alpha_constraint)
        
        # Initializers for Q
        self.beta_initializer = [initializers.Constant(value=beta_initializer[i]) for i in range(len(beta_initializer))]
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        
        if shared_axes is None:
            self.shared_axes = None
        elif not isinstance(shared_axes, (list, tuple)):
            self.shared_axes = [shared_axes]
        else:
            self.shared_axes = list(shared_axes)

    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        if self.shared_axes is not None:
            for i in self.shared_axes:
                param_shape[i - 1] = 1
        
        self.coeffsP = []
        for i in range(self.degreeP+1):
            # Add weight
            alpha_i = self.add_weight(shape=param_shape,
                                     name='alpha_%s'%i,
                                     initializer=self.alpha_initializer[i],
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
            # alpha_i = self.add_weight(shape=param_shape,
            #                          name='alpha_%s'%i,
            #                          initializer=self.alpha_initializer[i])
            self.coeffsP.append(alpha_i)
            
        # Create coefficients of Q
        self.coeffsQ = []
        for i in range(self.degreeQ+1):
            # Add weight
            beta_i = self.add_weight(shape=param_shape,
                                     name='beta_%s'%i,
                                     initializer=self.beta_initializer[i],
                                     regularizer=self.beta_regularizer,
                                     constraint=self.beta_constraint)
            # beta_i = self.add_weight(shape=param_shape,
            #                          name='beta_%s'%i,
            #                          initializer=self.beta_initializer[i])
            self.coeffsQ.append(beta_i)
        
        # Set input spec
        axes = {}
        if self.shared_axes:
            for i in range(1, len(input_shape)):
                if i not in self.shared_axes:
                    axes[i] = input_shape[i]
                    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
                    self.built = True

    def call(self, inputs, mask=None):
        # Evaluation of P
        outP = tf.math.polyval(self.coeffsP, inputs)
        # Evaluation of Q
        outQ = tf.math.polyval(self.coeffsQ, inputs)
        # Compute P/Q
        out = tf.math.divide(outP, outQ)
        return out

    def get_config(self):
        config = {
            'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'alpha_constraint': constraints.serialize(self.alpha_constraint),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'shared_axes': self.shared_axes
        }
        base_config = super(CustomDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape