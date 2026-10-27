import AdvGAN_Discriminator
import AdvGAN_Generator
import numpy as np
import pandas as pd
import tensorflow as tf

from petsard.synthesizer.synthesizer_base import BaseSynthesizer


class AdvGANSynthesizer(BaseSynthesizer):

    def __init__(self, config: dict, metadata):
        """
        初始化合成器。
        參數：
            config   : dict，來自 YAML 的所有配置參數
            metadata : SchemaMetadata，資料表的 Schema 資訊
        """


        super().__init__(config, metadata)

        self.thresh = 10.0
        self.mse = tf.keras.losses.MeanSquaredError()
        self.alpha = 10.0
        self.beta = 1.0
        self.lr = 0.001

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        pass

    def _fit(self, data: pd.DataFrame):
        """
        從輸入資料學習模式（訓練階段）。
        參數：
            data : pd.DataFrame，已前處理的訓練資料
                   （類別欄位已編碼，數值欄位已正規化）
        回傳：無（void）
        """
        self._data = data.copy(deep=True)
        self.feature_dim = len(self._data.columns)
        self.generator = AdvGAN_Generator.build_generator(feature_dim=self.feature_dim)
        self.discriminator = AdvGAN_Discriminator.build_discriminator(feature_dim=self.feature_dim)
        self.data_max = self._data.max()
        self.data_min = self._data.min()

        training_data = self._data.to_numpy()

        #for input_index in range(0, len(training_data)):
        for input_index in range(0, 10):
            target_training_data = training_data[input_index:input_index + 1]

            self.train_step(target_training_data)

        self._impl = True

    def _sample(self) -> pd.DataFrame:
        """
        生成合成資料（取樣階段）。
        參數：無
        回傳：
            pd.DataFrame，合成資料
            - 欄位名稱必須與 fit() 收到的 data 完全一致
            - 欄位順序應一致；資料型別應盡量與原始資料相符
        """


        training_data = self._data.to_numpy()
        generated_data_list = np.empty_like(training_data[0:1])

        #for input_index in range(0, len(training_data)):
        for input_index in range(0, 10):
            target_training_data = training_data[input_index:input_index + 1]

            perturbation = self.generator(target_training_data)
            generated_data = target_training_data + perturbation
            generated_data = tf.clip_by_value(generated_data, self.data_min, self.data_max)
            #generated_data = tf.maximum(target_training_data + perturbation, -1)
            #generated_data = tf.minimum(generated_data, 1)

            generated_data_list = np.vstack((generated_data_list, generated_data))

        generated_data_list = generated_data_list[1:]
        generated_data_list = pd.DataFrame(generated_data_list, columns = self._data.columns)

        return generated_data_list


    def perturb_loss(self, perturb, thresh=3.0):
        norm = tf.norm(perturb, axis=1)
        return tf.reduce_mean(tf.maximum(0.0, norm - thresh))


    def train_step(self, target_training_data):
         with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            perturbation = self.generator(target_training_data)
            generated_data = target_training_data + perturbation
            generated_data = tf.clip_by_value(generated_data, self.data_min, self.data_max)
            #generated_data = tf.maximum(target_training_data + perturbation, -1)
            #generated_data = tf.minimum(generated_data, 1)

            d_origin_logits, d_origin_probs = self.discriminator(target_training_data)
            d_perturbed_logits, d_perturbed_probs = self.discriminator(generated_data)

            l_perturb = self.perturb_loss(perturbation, self.thresh)
            g_loss_perturb = self.mse(tf.zeros_like(d_perturbed_probs), d_perturbed_probs)

            gen_loss = self.alpha * g_loss_perturb + self.beta * l_perturb

            d_loss_origin = self.mse(tf.ones_like(d_origin_probs), d_origin_probs)
            d_loss_perturb = self.mse(tf.zeros_like(d_perturbed_probs), d_perturbed_probs)
            disc_loss = d_loss_origin + d_loss_perturb


            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables, strict=False))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables, strict=False))
