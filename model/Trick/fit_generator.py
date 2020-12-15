def generator(input_ids,attention_mask,send_s_po,start_tokens,end_tokens,c_relation,batch_size):
    i=0
    while 1:
        input_ids_b = input_ids[i*batch_size:(i+1)*batch_size]
        attention_mask_b = attention_mask[i*batch_size:(i+1)*batch_size]
        send_s_po_b = send_s_po[i*batch_size:(i+1)*batch_size]
        start_tokens_b = start_tokens[i*batch_size:(i+1)*batch_size]
        end_tokens_b = end_tokens[i*batch_size:(i+1)*batch_size]
        c_relation_b = c_relation[i*batch_size:(i+1)*batch_size]
        # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
        yield({'input_1': input_ids_b, 'input_2': attention_mask_b,'input_3':send_s_po_b}, 
              {'s_start': start_tokens_b,'s_end':end_tokens_b,'relation':c_relation_b})
        i = (i+1)%(len(input_ids)//batch_size)
        
model.fit_generator(generator(input_ids,attention_mask,send_s_po,start_tokens,end_tokens,c_relation,batch_size),epochs=eopch,steps_per_epoch=steps_per_epoch,verbose=1,
                       callbacks=[Metrics(model_2,model_3,id2p,va_text_list,va_spo_list,va_input_ids,va_attention_mask,tokenizer)])
