#for tf.2.2+
def creat_FGM(epsilon=1.0):
    @tf.function 
    def train_step(self, data):
    '''
    计算在embedding上的gradient
    计算扰动 在embedding上加上扰动
    重新计算loss和gradient
    删除embedding上的扰动，并更新参数
    '''
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
	    with tf.GradientTape() as tape:
	        y_pred = model(x,training=True)
	        loss = loss_func(y,y_pred)
	    embedding = model.trainable_variables[0]
	    embedding_gradients = tape.gradient(loss,[model.trainable_variables[0]])[0]
	    embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
	    delta = 0.2 * embedding_gradients / (tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + 1e-8)  # 计算扰动
	    model.trainable_variables[0].assign_add(delta)
	    with tf.GradientTape() as tape2:
	        y_pred = model(x,training=True)
	        new_loss = loss_func(y,y_pred)
	    gradients = tape2.gradient(new_loss,model.trainable_variables)
	    model.trainable_variables[0].assign_sub(delta)
	    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
	    train_loss.update_state(loss)
	    return {m.name: m.result() for m in self.metrics}
    return train_step

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['acc'],)

#替换model.train_step 方法即可,并且删除原有的 train_function方法
model.train_step = functools.partial(train_step, model)
model.train_function = None

history = model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test),verbose=1,batch_size=32)



#for tf.2.2-
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_func = tf.losses.SparseCategoricalCrossentropy()
train_loss = tf.metrics.Mean(name='train_loss')

ds_train = tf.data.Dataset.from_tensor_slices((X_train,y_train)) \
          .shuffle(buffer_size = 1000).batch(32) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()
          
@tf.function
def train_step(model,x,y,loss_func,optimizer,train_loss):
    with tf.GradientTape() as tape:
        y_pred = model(x,training=True)
        loss = loss_func(y,y_pred)
    embedding = model.trainable_variables[0]
    embedding_gradients = tape.gradient(loss,[model.trainable_variables[0]])[0]
    embedding_gradients = tf.zeros_like(embedding) + embedding_gradients
    delta = 0.2 * embedding_gradients / (tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + 1e-8)  # 计算扰动
    model.trainable_variables[0].assign_add(delta)
    with tf.GradientTape() as tape2:
        y_pred = model(x,training=True)
        new_loss = loss_func(y,y_pred)
    gradients = tape2.gradient(new_loss,model.trainable_variables)
    model.trainable_variables[0].assign_sub(delta)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss.update_state(loss)

@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts%(24*60*60)
    hour = tf.cast(today_ts//3600+8,tf.int32)%tf.constant(24)
    minite = tf.cast((today_ts%3600)//60,tf.int32)
    second = tf.cast(tf.floor(today_ts%60),tf.int32)
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m))==1:
            return(tf.strings.format("0{}",m))
        else:
            return(tf.strings.format("{}",m))
    timestring = tf.strings.join([timeformat(hour),timeformat(minite),
                timeformat(second)],separator = ":")
    tf.print("=========="*8,end = "")
    tf.print(timestring)
    
def train_model(model,ds_train,epochs):
    for epoch in tf.range(1,epochs+1):
 
        for x, y in ds_train:
            train_step(model,x,y,loss_func,optimizer,train_loss)
 
        logs = 'Epoch={},Loss:{}'
        if epoch%1 ==0:
            printbar()
            tf.print(tf.strings.format(logs,(epoch,train_loss.result())))
            tf.print("")
        train_loss.reset_states()
        
train_model(model,ds_train,10)

