from vllm.core.scheduler import * 

from vllm.core.andes_utils.knapsack_solver import KnapSack

class AndesScheduler(Scheduler):
    """
    Implementation of the Andes, which is a QoE-aware serving system.
    Paper: https://arxiv.org/abs/2404.16283
    """
    def __init__(self, 
                scheduler_config: SchedulerConfig,
                cache_config: CacheConfig,
                lora_config: Optional[LoRAConfig]) -> None:
        super().__init__(scheduler_config, cache_config, lora_config)
        self.num_total_gpu_blocks = self.cache_config.num_gpu_blocks
        self.schedule_solver = KnapSack(self.cache_config.block_size,
                                        self.num_total_gpu_blocks,
                                        'greedy')
        
        
    def _schedule(self) -> SchedulerOutputs:
        return self._schedule_qoe_aware()

    def _schedule_qoe_aware(self) -> SchedulerOutputs:
        
        budget = SchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_seqs=self.scheduler_config.max_num_seqs,
        )
        enable_chunking = True
        curr_loras: Set[int] = set()

        # remaining_running, running_scheduled = (
        #     self.running, SchedulerRunningOutputs.create_empty())
        # remaining_waiting, prefills = (self.waiting,
        #                                SchedulerPrefillOutputs.create_empty())
        # remaining_swapped, swapped_in = (
        #     self.swapped, SchedulerSwappedInOutputs.create_empty())

        num_free_blocks = max(0, self.block_manager.gpu_allocator.get_num_free_blocks() - len(self.running))
        utilization = (self.num_total_gpu_blocks - num_free_blocks) / self.num_total_gpu_blocks

        # TODO: profile when starting the instance
        latency_function = None
        
        seq_to_admit, seq_to_swap_in, seq_to_evict = self.schedule_solver.schedule_requests(budget, self.running, self.waiting, self.swapped, utilization, latency_function)
        logger.info("Admitting %d, swapping in %d, evicting %d, Running %d, waiting %d, swapped %d", len(seq_to_admit), len(seq_to_swap_in), len(seq_to_evict), len(self.running), len(self.waiting), len(self.swapped))
        
        blocks_to_swap_out: Dict[int, int] = {}
        blocks_to_copy: Dict[int, List[int]] = {}
        blocks_to_swap_in: Dict[int, int] = {}

        seq_preempted: List[SequenceGroup] = []
        seq_swapped_out: List[SequenceGroup] = []

        # 1. evict requests to make room for new requests
        for seq_group in seq_to_evict:
            preempted_mode = self._preempt(seq_group,
                                        blocks_to_swap_out,
                                        PreemptionMode.SWAP)
            if preempted_mode == PreemptionMode.RECOMPUTE:
                seq_preempted.append(seq_group)
            else:
                seq_swapped_out.append(seq_group)

        # 2. Schedule the running requests 
        prev_running = list(set(self.running) - set(seq_preempted) - set(seq_swapped_out))
        scheduled_prefill_running: List[ScheduledSequenceGroup] = []
        scheduled_decode_running: List[ScheduledSequenceGroup] = []

        while prev_running:
            seq_group = prev_running[0]
            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                break

            prev_running.pop(0)
            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.remove(seq_group.lora_int_id)

                # TODO: may remove this overflow check
                if prev_running:
                    # Preempt the lowest-priority sequence groups.
                    victim_seq_group = prev_running.pop()
                    preempted_mode = self._preempt(victim_seq_group,
                                                   blocks_to_swap_out,
                                                   PreemptionMode.SWAP)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        seq_preempted.append(victim_seq_group)
                    else:
                        seq_swapped_out.append(victim_seq_group)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out,
                                                   PreemptionMode.SWAP)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        seq_preempted.append(seq_group)
                    else:
                        seq_swapped_out.append(seq_group)
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    scheduled_prefill_running.append(
                        ScheduledSequenceGroup(
                            seq_group=seq_group,
                            token_chunk_size=num_running_tokens))
                else:
                    scheduled_decode_running.append(
                        ScheduledSequenceGroup(seq_group=seq_group,
                                               token_chunk_size=1))
                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)
                if curr_loras is not None and seq_group.lora_int_id > 0:
                    curr_loras.add(seq_group.lora_int_id)


        # 3. admit 'new' requests
        ignored_seq_groups: List[SequenceGroup] = []
        leftover_waiting_sequences: Deque[SequenceGroup] = deque()
        admitted_seq: List[ScheduledSequenceGroup] = []

        while seq_to_admit: # and self._passed_delay(time.time()) :
            seq_group = seq_to_admit[0]
            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            if num_new_tokens > self.prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens,
                    self.prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                seq_to_admit.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_seq_groups.append(seq_group)
                seq_to_admit.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (self.lora_enabled and lora_int_id > 0
                        and lora_int_id not in curr_loras
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_waiting_sequences.appendleft(seq_group)
                    seq_to_admit.popleft()
                    continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            if curr_loras is not None and lora_int_id > 0:
                curr_loras.add(lora_int_id)
            seq_to_admit.popleft()
            self._allocate_and_set_running(seq_group)
            # To be scheduled
            admitted_seq.append(
                ScheduledSequenceGroup(seq_group=seq_group,
                                       token_chunk_size=num_new_tokens))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)
        
        if len(admitted_seq) > 0:
            self.prev_prompt = True

        # 4. swap in requests 
        infeasible_seq_groups: List[SequenceGroup] = []
        leftover_swapped: Deque[SequenceGroup] = deque()
        swappedin_decode_seq: List[ScheduledSequenceGroup] = []
        swappedin_prefill_seq: List[ScheduledSequenceGroup] = []
        
        while seq_to_swap_in:
            seq_group = seq_to_swap_in[0]
            alloc_status = self.block_manager.can_swap_in(seq_group)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_seq_groups.append(seq_group)
                seq_to_swap_in.popleft()
                continue

            lora_int_id = 0
            if self.lora_enabled:
                lora_int_id = seq_group.lora_int_id
                assert curr_loras is not None
                assert self.lora_config is not None
                if (lora_int_id > 0 and (lora_int_id not in curr_loras)
                        and len(curr_loras) >= self.lora_config.max_loras):
                    # We don't have a space for another LoRA, so
                    # we ignore this request for now.
                    leftover_swapped.appendleft(seq_group)
                    seq_to_swap_in.popleft()
                    continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            if lora_int_id > 0 and curr_loras is not None:
                curr_loras.add(lora_int_id)
            seq_to_swap_in.popleft()
            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()
            if is_prefill:
                swappedin_prefill_seq.append(
                    ScheduledSequenceGroup(seq_group=seq_group,
                                           token_chunk_size=num_new_tokens))
            else:
                swappedin_decode_seq.append(
                    ScheduledSequenceGroup(seq_group=seq_group, token_chunk_size=1))
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        # Update waiting requests
        self.waiting = [x for x in self.waiting if x not in [s.seq_group for s in admitted_seq]]
        self.waiting.extend(leftover_waiting_sequences)

        # Update swapped requests
        self.swapped = [x for x in self.swapped if x not in [s.seq_group for s in swappedin_prefill_seq] and x not in [s.seq_group for s in swappedin_decode_seq]]
        self.swapped.extend(leftover_swapped)
        self.swapped.extend(seq_swapped_out)
 
        # Update running requests
        self.running = prev_running 
        self.running.extend([s.seq_group for s in scheduled_prefill_running])
        self.running.extend([s.seq_group for s in scheduled_decode_running])
        self.running.extend([s.seq_group for s in admitted_seq])
        self.running.extend([s.seq_group for s in swappedin_decode_seq])  
        self.running.extend([s.seq_group for s in swappedin_prefill_seq])      
        
        num_lookahead_slots = self._get_num_lookahead_slots(is_prefill=False)
        
        return SchedulerOutputs(
            scheduled_seq_groups=(admitted_seq + scheduled_prefill_running + swappedin_prefill_seq + \
                                  scheduled_decode_running + swappedin_decode_seq ),
            num_prefill_groups=(len(admitted_seq) +
                                len(swappedin_prefill_seq) +
                                len(scheduled_prefill_running)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_swap_out=blocks_to_swap_out,
            blocks_to_copy=blocks_to_copy,
            ignored_seq_groups=ignored_seq_groups,
            num_lookahead_slots=num_lookahead_slots,
            running_queue_size=len(self.running),
        )

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
        preemption_mode: Optional[PreemptionMode] = None,
    ) -> PreemptionMode: 
        if preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP
        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            preemption_mode = self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        seq_group.preempt_signal()
        return preemption_mode

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if self._swap_out(seq_group, blocks_to_swap_out):
            return PreemptionMode.SWAP
        else:
            self._preempt_by_recompute(seq_group)
            return PreemptionMode.RECOMPUTE

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: Dict[int, int],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            # FIXME(woosuk): Abort the sequence group instead of aborting the
            # entire engine.
            logger.warning(
                "Recompute due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
            return False
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.update(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED
        return True
