import torch
import time
import statistics
import tracemalloc
from tqdm import tqdm

NUM_PER_CHUNK=60

@torch.no_grad()
def benchmark(
    model: torch.nn.Module,
    batch_size:int = 1,
    num_objects:int = 80,
    num_interactions:int = 117,
    num_instance_pairs=100,
    runs: int = 40,
    throw_out: float = 0.25,
    verbose: bool = True,
    device  = "cuda",
    reasoning_level='image',
    args=None
    # dtype = torch.bfloat16,
) -> float:
    model = model.eval().to(device)
    warm_up = int(runs * throw_out)
    # inputs = dataGenerator(case, batch_size=batch_size, n_page=n_page, token_enc=token_enc, token_dec=token_dec, layer_len=28, shape=shape, device=device, dtype=dtype)

    total, times, peak_memories = 0, [], []
    model.precomputed_text_embeddings = None
    images = torch.randn(batch_size, 3, args.input_resolution, args.input_resolution).to(device)

    if args.ml_decoder_query_type=='triplet':
        texts = ["a photo of a cat"] * (num_interactions * num_objects)
    else:
        texts = ["a photo of a cat"] * num_interactions
    inputs = {
        'images': images,
        'texts': texts,
    }
    
    if reasoning_level == 'instance':
        assert batch_size ==1
        # attention_mask = None
        so_indices = torch.zeros(num_instance_pairs).to(device=device, dtype=torch.long)
        inputs['so_indices'] = so_indices
        if model.attention_pooling.query_type == 'triplet' or args.use_union_cropped_image:
            model.attention_pooling.image_level_pooling = False
            images = torch.randn(num_instance_pairs, 3, args.input_resolution, args.input_resolution).to(device)
            inputs['images'] = images
        elif model.attention_pooling.query_type == 'object':
            model.attention_pooling.instance_score_scheme = 'so_region'
            model.attention_pooling.instance_score_post_masking_type = 'post_sum'
            model.attention_pooling.post_sum_scale = 1.0
            model.attention_pooling.mask_generation_lib = 'torch' # 'numpy'
            model.use_det_results = True
            dummy_boxes = [0,0,518,518]
            meta_data = [{'union_boxes':torch.tensor([dummy_boxes] * num_instance_pairs).to(device),
                          'human_boxes':torch.tensor([dummy_boxes] * num_instance_pairs).to(device),
                          'object_boxes':torch.tensor([dummy_boxes] * num_instance_pairs).to(device),
                          'input_size':torch.tensor([args.input_resolution, args.input_resolution]).to(device)}]
            inputs['meta_data'] = meta_data
                
    # inputs = torch.randn()
    for i in tqdm(range(runs), disable=not verbose, desc="Benchmarking"):
        if i == warm_up:
            start, total, times, peak_memories = time.time(), 0, [], []

        tracemalloc.start()
        start_gpt = time.perf_counter()
        # with torch.cuda.amp.autocast(enabled=True, dtype=dtype):
        if reasoning_level == 'instance' and model.attention_pooling.query_type == 'triplet':
            chunked_images = torch.tensor_split(inputs['images'], inputs['images'].shape[0] // NUM_PER_CHUNK + 1)
            chunked_so_indices = torch.tensor_split(inputs['so_indices'], inputs['so_indices'].shape[0] // NUM_PER_CHUNK + 1)
            new_inputs = {}
            for image, so_indices in zip(chunked_images, chunked_so_indices):
                new_inputs['images'] = image
                new_inputs['so_indices'] = so_indices
                new_inputs['texts'] = texts
                model(**new_inputs)
        else:
            model(**inputs)
        end_gpt = time.perf_counter()
        total += batch_size

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        times.append(end_gpt - start_gpt)
        peak_memories.append(peak / 1024)  # KB로 변환

    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times)
    avg_memory = statistics.mean(peak_memories)
    std_memory = statistics.stdev(peak_memories)

    end = time.time()
    elapsed = end - start
    throughput = total / elapsed
    if verbose:
        print(f"🔁 Benchmark over {int(runs * (1 - throw_out))} runs:")
        print(f"⏱️ Time (s):   {avg_time:.6f} sec (± {std_time:.6f})")
        print(f"⏱️ Time (ms):   {avg_time * 1000:.2f} ms (± {std_time * 1000:.2f} ms)")
        print(f"📦 Memory (KB): {avg_memory:.2f} KB (± {std_memory:.2f})")
        print(f"📦 Memory (MB): {avg_memory / 1024:.2f} MB (± {std_memory / 1024:.2f} MB)")
        print(f"Throughput: {throughput:.2f} im/s")
        print()

    # Prepare benchmark results
    benchmark_results = {
        'num_objects': num_objects,
        'num_interactions': num_interactions,
        'batch_size': batch_size,
        'runs': runs,
        'throw_out': throw_out,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'avg_memory_mb': avg_memory / 1024,
        'std_memory_mb': std_memory / 1024,
        'throughput_im_per_sec': throughput
    }

    return benchmark_results