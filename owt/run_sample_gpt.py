import os
import argparse
import torch
import torch.multiprocessing as mp
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def save_text_to_file(text, file_path):
    with open(file_path, 'w') as file:
        file.write(text)

def generate_fixed_length_text_batch(model, tokenizer, batch_size=1, max_length=1024, min_length=1024, top_p=0.8, temp=1.0):
    input_ids = tokenizer.encode(tokenizer.bos_token, return_tensors='pt').to('cuda').repeat(batch_size, 1)
    eos_token_id = tokenizer.eos_token_id

    generated_ids = input_ids
    print(top_p, temp)
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            inputs=input_ids,
            max_length=max_length+1, # +1 to ignore the BOS token
            min_length=min_length+1,
            do_sample=True,
            top_p=top_p,
            top_k=0,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_texts, generated_ids[:, 1:]

def load_model_and_tokenizer(model_id, cache_dir):
    tokenizer = GPT2Tokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.padding_side = 'left'
    model = GPT2LMHeadModel.from_pretrained(model_id, cache_dir=cache_dir)
    return model, tokenizer

def generate_and_save_texts_on_gpu(gpu_id, start_index, end_index, output_directory, model_id, cache_dir, batch_size, max_length, min_length, top_p, temp):
    torch.cuda.set_device(gpu_id)
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)
    model.to(f'cuda:{gpu_id}')
    
    num_batches = (end_index - start_index + batch_size - 1) // batch_size
    generated_ids_all = []

    for batch in range(num_batches):
        current_batch_size = min(batch_size, end_index - start_index - batch * batch_size)
        generated_texts, generated_ids = generate_fixed_length_text_batch(model, tokenizer, batch_size=current_batch_size, max_length=max_length, min_length=min_length, top_p=top_p, temp=temp)
        
        for i, text in enumerate(generated_texts):
            index = start_index + batch * batch_size + i
            file_path = os.path.join(output_directory, f'samples_{index}.txt')
            generated_ids_all.append(generated_ids[i])
            save_text_to_file(text, file_path)
            print(f'GPU {gpu_id} - Saved: {file_path}')
    
    all_ids = torch.stack(generated_ids_all)
    print(torch.min(all_ids - generated_ids), torch.max(all_ids - generated_ids))
    print(f"shape of saved ids {all_ids.shape}")
    torch.save(all_ids, os.path.join(output_directory, f'generated_ids_{start_index}_{end_index}.pt'))

def main():
    parser = argparse.ArgumentParser(description="Generate some samples")
    parser.add_argument("--model_id", default='gpt2-medium', type=str, help="Specify your model ID")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p value for nucleus sampling")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--allow_eos", action='store_true', help="Allow end of sequence (default: False)")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--num_texts_per_gpu", type=int, default=20, help="Number of texts to generate per GPU")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of generated sequences")
    parser.add_argument("--cache_dir", type=str, default="/pscratch/sd/s/sulinl/gpt_cache", help="Cache directory for model and tokenizer")
    args = parser.parse_args()

    
    model_id_parts = args.model_id.split('/')
    model_id_last_part = model_id_parts[-1]
    output_directory = f'/pscratch/sd/s/sulinl/generated_texts/gen-{model_id_last_part}/p_{args.top_p}_t_{args.temp}_eos_{args.allow_eos}'
    os.makedirs(output_directory, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    if args.allow_eos:
        min_length = 0
    else:
        min_length = args.max_length
    processes = []
    for gpu_id in range(num_gpus):
        start_index = gpu_id * args.num_texts_per_gpu
        end_index = start_index + args.num_texts_per_gpu
        p = mp.Process(target=generate_and_save_texts_on_gpu, args=(
            gpu_id, start_index, end_index, output_directory, args.model_id, args.cache_dir, args.batch_size, args.max_length, min_length, args.top_p, args.temp))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
