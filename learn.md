
---
12/22
1. I'm interested in, the training data of this model, basically the inputs and target, what are their shape? batch_size, seq_len?
2. when training, i know the attn computation is complex, there are some special attention mask stuff, i want to know how this stuff is set, find all related variable, data structure that control this part, also, i think this part is set before the model start to train, i.e. before `logits = model(input, target) 
So, lets first start on the surface, i.e. what files we should has a glimpse, this is definitely now a one step work, we just plan for first step


 you do the task again, we focus on bd3lm this model, we focus on the senario that the model in trained in a "block" unit and there should be input to the model twice of the original seq_len, and the last half is for computing the pure kv for every first half block to attn to since every block in the first half is "noised"
