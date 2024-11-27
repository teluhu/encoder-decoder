# encoder-decoder
Reproduction of the transformer architecture, like the encoder and decoder. 


# encoder部分
下列任务ddl：2024-12-01

11.23更新：原来Input Embedding和positional_encoding不算在encoder里，因为这两层不参与encoder的前向传播。

### 任务列表
**学习并实现Input Embedding** 
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.23
   - **状态**：
         ✅已完成
     
         学习了Input Embedding实现方法

         完成了Input Embedding实现，即调用nn.Embedding函数即可
     
         ⏳ 暂时放在了Transformer的class里
     
**学习并实现Positional_encoding**  
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.26
   - **状态**：         ✅已完成
     
         学习了Positional Embedding实现方法

         完成了Positional Embedding实现，输入维度是[batch_size, length, dmodel]

         ⚙️ 修改了公式的实现，现在dmodel也可以是奇数了
     
         ⏳ 暂时放在了Transformer的class里

**学习并实现多头注意力编码**  
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.27
   - **状态**：         ✅已完成
     
         学习了多头注意力实现方法

         完成了Positional Embedding实现，输入维度是[batch_size, length, dmodel]，输出维度是[batch_size, length, dmodel]

         ⚙️ 把concat里的.contiguous().view改成了.reshape()
     
         ❗ 把mask的部分省略了
     
         ⏳ 暂时放在了Encoder的class里

**学习并实现 Add & Norm**  
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.27
   - **状态**：         ✅已完成
     
         学习了LayerNorm的实现方式

         区别了BatchNorm1d和LayerNorm的区别
     
         ⏳ 暂时放在了Encoder的class里
     
**学习并实现 Feed Forward**  
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.27
   - **状态**：         ✅已完成
     
         学习了FeedForward的实现方式

         学习了FeedForward的存在意义

         ⚙️ 把FeedForward里加了dropout层，如果用的话，is_drop传入True即可
     
         ⏳ 暂时放在了Encoder的class里

**整合encoder部分**  
   - **负责人**：zrz  
   - **计划完成时间**：2024.11.27
   - **状态**：         ✅已完成
     
         普通地搭起来了
         ⏳不过参考的博客在attn后加了dropout；我在前馈神经网络中已经加了dropout，就不再加了，看看效果 

# decoder部分

### 任务列表
**学习并实现outputs Embedding** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**学习并实现positional encoding** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**学习并实现masked multi-head attention** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**学习并实现add & norm** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**实现multi-head attention并整合encoder** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**学习并实现feed forward** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**学习并实现linear + softmax** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
**decoder** 
   - **负责人**：hjy 
   - **计划完成时间**：xxx
   - **状态**：未完成
     
![5c40047bf06cb2f9603d5ef42921c43](https://github.com/user-attachments/assets/ecd15048-1c42-482b-a9f0-413ba023fdf2)
