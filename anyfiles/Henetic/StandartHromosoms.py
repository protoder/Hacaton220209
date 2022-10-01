UNet0 = (   9999999999999999, 9999999999999999, # 1-st поля хромосомы служебные: Признак измененности, ID
                4, 1, 252, 0, 0.01156, 252, 252, 252, 252,# заголовок сети
                1, 252, 252, 8, 8,# описание выходного блока
                3, 3, 2, # загоовок блока 0
                16, len(activation_list), 2, 252, 252, 252, 8, 8,
                16, len(activation_list), 2, 252, 252, 252, 8, 8,
                16, len(activation_list), 2, 252, 252, 252, 8, 8,
                16, len(activation_list), 2, 252, 252, 252, 8, 8,
                3, 3, 2,  # загоовок блока 1
                32, len(activation_list), 2, 252, 252, 252, 8, 8,
                32, len(activation_list), 2, 252, 252, 252, 8, 8,
                32, len(activation_list), 2, 252, 252, 252, 8, 8,
                32, len(activation_list), 2, 252, 252, 252, 8, 8,
                3, 3, 2,  # загоовок блока 2
                64, len(activation_list), 2, 252, 252, 252, 8, 8,
                64, len(activation_list), 2, 252, 252, 252, 8, 8,
                64, len(activation_list), 2, 252, 252, 252, 8, 8,
                64, len(activation_list), 2, 252, 252, 252, 8, 8,
                3, 3, 2,  # загоовок блока 3
                128, len(activation_list), 2, 252, 252, 252, 8, 8,
                128, len(activation_list), 2, 252, 252, 252, 8, 8,
                128, len(activation_list), 2, 252, 252, 252, 8, 8,
                128, len(activation_list), 2, 252, 252, 252, 8, 8,


                            # Дальше пошли блоки энкодера

                            3,   # загоовок блока декодера 3
                            128, len(activation_list), 2, 252, 252, 252,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            128, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 2
                            64, len(activation_list), 2, 252, 252, 252,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            64, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 1
                            32, len(activation_list), 2, 252, 252, 252,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            32, len(activation_list), 2, 252, 252, 252, 8, 8,
                            3,   # загоовок блока декодера 0
                            16, len(activation_list), 2, 252, 252, 252,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8,
                            16, len(activation_list), 2, 252, 252, 252, 8, 8)
