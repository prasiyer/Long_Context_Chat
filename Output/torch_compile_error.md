Running experiment:  torch.compile
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:None for open-end generation.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
---------------------------------------------------------------------------
Unsupported                               Traceback (most recent call last)
Cell In[16], line 82
     80         outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)
     81 else:
---> 82         outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)  
     83 # outputs = model.generate(input_ids, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6, top_p=0.9,)
     84 end_time = time.time()

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/utils/_contextlib.py:116, in context_decorator.<locals>.decorate_context(*args, **kwargs)
    113 @functools.wraps(func)
    114 def decorate_context(*args, **kwargs):
    115     with ctx_factory():
--> 116         return func(*args, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/transformers/generation/utils.py:2215, in GenerationMixin.generate(self, inputs, generation_config, logits_processor, stopping_criteria, prefix_allowed_tokens_fn, synced_gpus, assistant_model, streamer, negative_prompt_ids, negative_prompt_attention_mask, **kwargs)
   2207     input_ids, model_kwargs = self._expand_inputs_for_generation(
   2208         input_ids=input_ids,
   2209         expand_size=generation_config.num_return_sequences,
   2210         is_encoder_decoder=self.config.is_encoder_decoder,
   2211         **model_kwargs,
   2212     )
   2214     # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
-> 2215     result = self._sample(
   2216         input_ids,
   2217         logits_processor=prepared_logits_processor,
   2218         stopping_criteria=prepared_stopping_criteria,
   2219         generation_config=generation_config,
   2220         synced_gpus=synced_gpus,
   2221         streamer=streamer,
   2222         **model_kwargs,
   2223     )
   2225 elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
   2226     # 11. prepare beam search scorer
   2227     beam_scorer = BeamSearchScorer(
   2228         batch_size=batch_size,
   2229         num_beams=generation_config.num_beams,
   (...)
   2234         max_length=generation_config.max_length,
   2235     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/transformers/generation/utils.py:3206, in GenerationMixin._sample(self, input_ids, logits_processor, stopping_criteria, generation_config, synced_gpus, streamer, **model_kwargs)
   3203 model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})
   3205 # forward pass to get next token
-> 3206 outputs = self(**model_inputs, return_dict=True)
   3208 # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
   3209 model_kwargs = self._update_model_kwargs_for_generation(
   3210     outputs,
   3211     model_kwargs,
   3212     is_encoder_decoder=self.config.is_encoder_decoder,
   3213 )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
   1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735 else:
-> 1736     return self._call_impl(*args, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
   1742 # If we don't have any hooks, we want to skip the rest of the logic in
   1743 # this function, and just call forward.
   1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
   1745         or _global_backward_pre_hooks or _global_backward_hooks
   1746         or _global_forward_hooks or _global_forward_pre_hooks):
-> 1747     return forward_call(*args, **kwargs)
   1749 result = None
   1750 called_always_called_hooks = set()

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/eval_frame.py:465, in _TorchDynamoContext.__call__.<locals>._fn(*args, **kwargs)
    460 saved_dynamic_layer_stack_depth = (
    461     torch._C._functorch.get_dynamic_layer_stack_depth()
    462 )
    464 try:
--> 465     return fn(*args, **kwargs)
    466 finally:
    467     # Restore the dynamic layer stack depth if necessary.
    468     torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
    469         saved_dynamic_layer_stack_depth
    470     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:1269, in CatchErrorsWrapper.__call__(self, frame, cache_entry, frame_state)
   1263             return hijacked_callback(
   1264                 frame, cache_entry, self.hooks, frame_state
   1265             )
   1267 with compile_lock, _disable_current_modes():
   1268     # skip=1: skip this frame
-> 1269     return self._torchdynamo_orig_callable(
   1270         frame, cache_entry, self.hooks, frame_state, skip=1
   1271     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:526, in ConvertFrameAssert.__call__(self, frame, cache_entry, hooks, frame_state, skip)
    510 compile_id = CompileId(frame_id, frame_compile_id)
    512 signpost_event(
    513     "dynamo",
    514     "_convert_frame_assert._compile",
   (...)
    523     },
    524 )
--> 526 return _compile(
    527     frame.f_code,
    528     frame.f_globals,
    529     frame.f_locals,
    530     frame.f_builtins,
    531     self._torchdynamo_orig_callable,
    532     self._one_graph,
    533     self._export,
    534     self._export_constraints,
    535     hooks,
    536     cache_entry,
    537     cache_size,
    538     frame,
    539     frame_state=frame_state,
    540     compile_id=compile_id,
    541     skip=skip + 1,
    542 )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:924, in _compile(code, globals, locals, builtins, compiler_fn, one_graph, export, export_constraints, hooks, cache_entry, cache_size, frame, frame_state, compile_id, skip)
    922 guarded_code = None
    923 try:
--> 924     guarded_code = compile_inner(code, one_graph, hooks, transform)
    925     return guarded_code
    926 except Exception as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:666, in _compile.<locals>.compile_inner(code, one_graph, hooks, transform)
    664 with dynamo_timed("_compile.compile_inner", phase_name="entire_frame_compile"):
    665     with CompileTimeInstructionCounter.record():
--> 666         return _compile_inner(code, one_graph, hooks, transform)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_utils_internal.py:87, in compile_time_strobelight_meta.<locals>.compile_time_strobelight_meta_inner.<locals>.wrapper_function(*args, **kwargs)
     84     kwargs["skip"] = kwargs["skip"] + 1
     86 if not StrobelightCompileTimeProfiler.enabled:
---> 87     return function(*args, **kwargs)
     89 return StrobelightCompileTimeProfiler.profile_compile_time(
     90     function, phase_name, *args, **kwargs
     91 )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:699, in _compile.<locals>._compile_inner(code, one_graph, hooks, transform)
    697 CompileContext.get().attempt = attempt
    698 try:
--> 699     out_code = transform_code_object(code, transform)
    700     break
    701 except exc.RestartAnalysis as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/bytecode_transformation.py:1322, in transform_code_object(code, transformations, safe)
   1319 instructions = cleaned_instructions(code, safe)
   1320 propagate_line_nums(instructions)
-> 1322 transformations(instructions, code_options)
   1323 return clean_and_assemble_instructions(instructions, keys, code_options)[1]

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:219, in preserve_global_state.<locals>._fn(*args, **kwargs)
    215 exit_stack.enter_context(
    216     torch.fx._symbolic_trace._maybe_revert_all_patches()
    217 )
    218 try:
--> 219     return fn(*args, **kwargs)
    220 finally:
    221     cleanup.close()

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/convert_frame.py:634, in _compile.<locals>.transform(instructions, code_options)
    632 try:
    633     with tracing(tracer.output.tracing_context), tracer.set_current_tx():
--> 634         tracer.run()
    635 except exc.UnspecializeRestartAnalysis:
    636     speculation_log.clear()

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2796, in InstructionTranslator.run(self)
   2795 def run(self):
-> 2796     super().run()

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:1680, in InstructionTranslatorBase.CALL_FUNCTION_EX(self, inst)
   1678 # Map to a dictionary of str -> VariableTracker
   1679 kwargsvars = kwargsvars.keys_as_python_constant()
-> 1680 self.call_function(fn, argsvars.items, kwargsvars)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/lazy.py:156, in _create_realize_and_forward.<locals>.realize_and_forward(self, *args, **kwargs)
    154 @functools.wraps(getattr(VariableTracker, name))
    155 def realize_and_forward(self, *args, **kwargs):
--> 156     return getattr(self.realize(), name)(*args, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:904, in FunctoolsPartialVariable.call_function(self, tx, args, kwargs)
    902 merged_args = self.args + args
    903 merged_kwargs = {**self.keywords, **kwargs}
--> 904 return self.func.call_function(tx, merged_args, merged_kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:1680, in InstructionTranslatorBase.CALL_FUNCTION_EX(self, inst)
   1678 # Map to a dictionary of str -> VariableTracker
   1679 kwargsvars = kwargsvars.keys_as_python_constant()
-> 1680 self.call_function(fn, argsvars.items, kwargsvars)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:385, in UserMethodVariable.call_function(self, tx, args, kwargs)
    383     fn = getattr(self.obj.value, self.fn.__name__)
    384     return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
--> 385 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2279, in InstructionTranslatorBase.CALL(self, inst)
   2277 @break_graph_if_unsupported(push=1)
   2278 def CALL(self, inst):
-> 2279     self._call(inst)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2273, in InstructionTranslatorBase._call(self, inst, call_kw)
   2268     kwargs = {}
   2270 try:
   2271     # if call_function fails, need to set kw_names to None, otherwise
   2272     # a subsequent call may have self.kw_names set to an old value
-> 2273     self.call_function(fn, args, kwargs)
   2274 finally:
   2275     self.kw_names = None

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/lazy.py:156, in _create_realize_and_forward.<locals>.realize_and_forward(self, *args, **kwargs)
    154 @functools.wraps(getattr(VariableTracker, name))
    155 def realize_and_forward(self, *args, **kwargs):
--> 156     return getattr(self.realize(), name)(*args, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/nn_module.py:899, in UnspecializedNNModuleVariable.call_function(self, tx, args, kwargs)
    891 ctx = (
    892     record_nn_module_stack(
    893         str(id(mod)), self.get_nn_module_stack_source(), tx, mod
   (...)
    896     else nullcontext()
    897 )
    898 with ctx:
--> 899     return variables.UserFunctionVariable(fn, source=source).call_function(
    900         tx, [self] + list(args), kwargs
    901     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2279, in InstructionTranslatorBase.CALL(self, inst)
   2277 @break_graph_if_unsupported(push=1)
   2278 def CALL(self, inst):
-> 2279     self._call(inst)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2273, in InstructionTranslatorBase._call(self, inst, call_kw)
   2268     kwargs = {}
   2270 try:
   2271     # if call_function fails, need to set kw_names to None, otherwise
   2272     # a subsequent call may have self.kw_names set to an old value
-> 2273     self.call_function(fn, args, kwargs)
   2274 finally:
   2275     self.kw_names = None

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/nn_module.py:899, in UnspecializedNNModuleVariable.call_function(self, tx, args, kwargs)
    891 ctx = (
    892     record_nn_module_stack(
    893         str(id(mod)), self.get_nn_module_stack_source(), tx, mod
   (...)
    896     else nullcontext()
    897 )
    898 with ctx:
--> 899     return variables.UserFunctionVariable(fn, source=source).call_function(
    900         tx, [self] + list(args), kwargs
    901     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:1680, in InstructionTranslatorBase.CALL_FUNCTION_EX(self, inst)
   1678 # Map to a dictionary of str -> VariableTracker
   1679 kwargsvars = kwargsvars.keys_as_python_constant()
-> 1680 self.call_function(fn, argsvars.items, kwargsvars)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/lazy.py:156, in _create_realize_and_forward.<locals>.realize_and_forward(self, *args, **kwargs)
    154 @functools.wraps(getattr(VariableTracker, name))
    155 def realize_and_forward(self, *args, **kwargs):
--> 156     return getattr(self.realize(), name)(*args, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:904, in FunctoolsPartialVariable.call_function(self, tx, args, kwargs)
    902 merged_args = self.args + args
    903 merged_kwargs = {**self.keywords, **kwargs}
--> 904 return self.func.call_function(tx, merged_args, merged_kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:1680, in InstructionTranslatorBase.CALL_FUNCTION_EX(self, inst)
   1678 # Map to a dictionary of str -> VariableTracker
   1679 kwargsvars = kwargsvars.keys_as_python_constant()
-> 1680 self.call_function(fn, argsvars.items, kwargsvars)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:385, in UserMethodVariable.call_function(self, tx, args, kwargs)
    383     fn = getattr(self.obj.value, self.fn.__name__)
    384     return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
--> 385 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2279, in InstructionTranslatorBase.CALL(self, inst)
   2277 @break_graph_if_unsupported(push=1)
   2278 def CALL(self, inst):
-> 2279     self._call(inst)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2273, in InstructionTranslatorBase._call(self, inst, call_kw)
   2268     kwargs = {}
   2270 try:
   2271     # if call_function fails, need to set kw_names to None, otherwise
   2272     # a subsequent call may have self.kw_names set to an old value
-> 2273     self.call_function(fn, args, kwargs)
   2274 finally:
   2275     self.kw_names = None

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2279, in InstructionTranslatorBase.CALL(self, inst)
   2277 @break_graph_if_unsupported(push=1)
   2278 def CALL(self, inst):
-> 2279     self._call(inst)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2273, in InstructionTranslatorBase._call(self, inst, call_kw)
   2268     kwargs = {}
   2270 try:
   2271     # if call_function fails, need to set kw_names to None, otherwise
   2272     # a subsequent call may have self.kw_names set to an old value
-> 2273     self.call_function(fn, args, kwargs)
   2274 finally:
   2275     self.kw_names = None

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

    [... skipping similar frames: InstructionTranslatorBase.CALL at line 2279 (1 times), InstructionTranslatorBase._call at line 2273 (1 times), BaseUserFunctionVariable.call_function at line 111 (1 times), InstructionTranslatorBase.call_function at line 830 (1 times), InliningInstructionTranslator.inline_call at line 3011 (1 times), InliningInstructionTranslator.inline_call_ at line 3139 (1 times), InstructionTranslatorBase.inline_user_function_return at line 836 (1 times), InstructionTranslatorBase.run at line 983 (1 times), InstructionTranslatorBase.step at line 895 (1 times), break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper at line 582 (1 times)]

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2279, in InstructionTranslatorBase.CALL(self, inst)
   2277 @break_graph_if_unsupported(push=1)
   2278 def CALL(self, inst):
-> 2279     self._call(inst)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:2273, in InstructionTranslatorBase._call(self, inst, call_kw)
   2268     kwargs = {}
   2270 try:
   2271     # if call_function fails, need to set kw_names to None, otherwise
   2272     # a subsequent call may have self.kw_names set to an old value
-> 2273     self.call_function(fn, args, kwargs)
   2274 finally:
   2275     self.kw_names = None

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:385, in UserMethodVariable.call_function(self, tx, args, kwargs)
    383     fn = getattr(self.obj.value, self.fn.__name__)
    384     return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
--> 385 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:324, in UserFunctionVariable.call_function(self, tx, args, kwargs)
    319 if self.is_constant:
    320     return invoke_and_store_as_constant(
    321         tx, self.fn, self.get_name(), args, kwargs
    322     )
--> 324 return super().call_function(tx, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/functions.py:111, in BaseUserFunctionVariable.call_function(self, tx, args, kwargs)
    105 def call_function(
    106     self,
    107     tx: "InstructionTranslator",
    108     args: "List[VariableTracker]",
    109     kwargs: "Dict[str, VariableTracker]",
    110 ) -> "VariableTracker":
--> 111     return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:836, in InstructionTranslatorBase.inline_user_function_return(self, fn, args, kwargs)
    832 def inline_user_function_return(self, fn, args, kwargs):
    833     """
    834     A call to some user defined function by inlining it.
    835     """
--> 836     return InliningInstructionTranslator.inline_call(self, fn, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3011, in InliningInstructionTranslator.inline_call(cls, parent, func, args, kwargs)
   3008 @classmethod
   3009 def inline_call(cls, parent, func, args, kwargs):
   3010     with patch.dict(counters, {"unimplemented": counters["inline_call"]}):
-> 3011         return cls.inline_call_(parent, func, args, kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:3139, in InliningInstructionTranslator.inline_call_(parent, func, args, kwargs)
   3137 try:
   3138     with strict_ctx:
-> 3139         tracer.run()
   3140 except exc.ObservedException as e:
   3141     msg = f"Observed exception DURING INLING {code} : {e}"

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:983, in InstructionTranslatorBase.run(self)
    981 try:
    982     self.output.push_tx(self)
--> 983     while self.step():
    984         pass
    985 except BackendCompilerFailed:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:895, in InstructionTranslatorBase.step(self)
    892 self.update_block_stack(inst)
    894 try:
--> 895     self.dispatch_table[inst.opcode](self, inst)
    896     return not self.output.should_exit
    897 except exc.ObservedException as e:

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:582, in break_graph_if_unsupported.<locals>.decorator.<locals>.wrapper(self, inst)
    580     return handle_graph_break(self, inst, speculation.reason)
    581 try:
--> 582     return inner_fn(self, inst)
    583 except Unsupported as excp:
    584     if self.generic_context_manager_depth > 0:
    585         # We don't support graph break under GenericContextWrappingVariable,
    586         # If there is, we roll back to the checkpoint and fall back.

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:1680, in InstructionTranslatorBase.CALL_FUNCTION_EX(self, inst)
   1678 # Map to a dictionary of str -> VariableTracker
   1679 kwargsvars = kwargsvars.keys_as_python_constant()
-> 1680 self.call_function(fn, argsvars.items, kwargsvars)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/symbolic_convert.py:830, in InstructionTranslatorBase.call_function(self, fn, args, kwargs)
    828 if inner_fn and callable(inner_fn) and is_forbidden(inner_fn):
    829     raise AssertionError(f"Attempt to trace forbidden callable {inner_fn}")
--> 830 self.push(fn.call_function(self, args, kwargs))

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/torch.py:897, in TorchInGraphFunctionVariable.call_function(self, tx, args, kwargs)
    888             if "out" in kwargs and isinstance(kwargs["out"], variables.TensorVariable):
    889                 # Calling fake tensor propagation can mutate the out= tensor in
    890                 # tx.output.tracked_fakes. tracked_fakes are used to apply
   (...)
    893                 # guards. So save the shape now, and check later if it has
    894                 # changed. If it has, graph break.
    895                 fake_out_shape = kwargs["out"].proxy.node.meta["example_value"].shape
--> 897             tensor_variable = wrap_fx_proxy(
    898                 tx=tx,
    899                 proxy=tx.output.create_proxy(
    900                     "call_function",
    901                     fn_,
    902                     *proxy_args_kwargs(args, kwargs),
    903                 ),
    904             )
    906             if (
    907                 isinstance(tensor_variable, TensorVariable)
    908                 and "requires_grad" in kwargs
    909                 and kwargs["requires_grad"].as_python_constant()
    910             ):
    911                 unimplemented(
    912                     """factory functions that return tensors that require grad are not supported.
    913 Either create the tensor outside the compiled region, or do not set the tensor to require_grad"""
    914                 )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/builder.py:2037, in wrap_fx_proxy(tx, proxy, example_value, subclass_type, **options)
   2029 kwargs = {
   2030     "tx": tx,
   2031     "proxy": proxy,
   (...)
   2034     **options,
   2035 }
   2036 if subclass_type is None:
-> 2037     return wrap_fx_proxy_cls(target_cls=TensorVariable, **kwargs)
   2038 else:
   2039     result = wrap_fx_proxy_cls(target_cls=TensorWithTFOverrideVariable, **kwargs)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/builder.py:2234, in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, subclass_type, **options)
   2230             options_i = options
   2232         # WARNING: this assumes the same target_cls as this tuple/list call
   2233         unpacked.append(
-> 2234             wrap_fx_proxy_cls(
   2235                 target_cls=target_cls,
   2236                 tx=tx,
   2237                 proxy=proxy_i,
   2238                 example_value=val,
   2239                 **options_i,
   2240             )
   2241         )
   2242 if isinstance(example_value, torch.Size):
   2243     # NB: Keep the old proxy around.  See SizeVariable for an
   2244     # explanation why
   2245     return SizeVariable(unpacked, proxy, **options)

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/variables/builder.py:2333, in wrap_fx_proxy_cls(target_cls, tx, proxy, example_value, subclass_type, **options)
   2331     return ConstantVariable.create(example_value, **options)
   2332 else:
-> 2333     unimplemented(
   2334         "torch.* op returned non-Tensor "
   2335         + f"{typestr(example_value)} {proxy.node.op} {proxy.node.target}",
   2336         case_name="unsupported_operator",
   2337     )

File /anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/exc.py:297, in unimplemented(msg, from_exc, case_name)
    295 if from_exc is not _NOTHING:
    296     raise Unsupported(msg, case_name=case_name) from from_exc
--> 297 raise Unsupported(msg, case_name=case_name)

Unsupported: torch.* op returned non-Tensor device call_function <built-in function getitem>

from user code:
   File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/_dynamo/external_utils.py", line 40, in inner
    return fn(*args, **kwargs)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/hooks.py", line 170, in new_forward
    output = module._old_forward(*args, **kwargs)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py", line 1190, in forward
    outputs = self.model(
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py", line 945, in forward
    layer_outputs = decoder_layer(
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
    return forward_call(*args, **kwargs)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/hooks.py", line 165, in new_forward
    args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/hooks.py", line 364, in pre_forward
    return send_to_device(args, self.execution_device), send_to_device(
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/utils/operations.py", line 184, in send_to_device
    {
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/utils/operations.py", line 185, in <dictcomp>
    k: t if k in skip_keys else send_to_device(t, device, non_blocking=non_blocking, skip_keys=skip_keys)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/accelerate/utils/operations.py", line 156, in send_to_device
    return tensor.to(device, non_blocking=non_blocking)
  File "/anaconda/envs/pi4_py311/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1299, in to
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(

Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True