import torch
from datasets import load_dataset
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import pandas as pd
import os
import math

def setup_model_and_processor():
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model.to(f"cuda:{torch.distributed.get_rank()}")
    return model, processor

def generate_new_caption(model, processor, image, original_caption):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": f"This image has the following caption: '{original_caption}'. Please provide a good and short description of photography of the fashion item in this image."
                },
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    
    try:
        model_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )

        if 'input_ids' not in model_inputs:
            raise ValueError("Processor did not generate input_ids")

        model_inputs = {k: v.to(model.device) for k, v in model_inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            generated_ids_trimmed = outputs[:, model_inputs['input_ids'].shape[1]:]
            
            new_caption = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

        return new_caption

    except Exception as e:
        print(f"Error in generate_new_caption: {str(e)}")
        raise e

def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")
    
    dataset = load_dataset("hahminlew/kream-product-blip-captions")
    train_dataset = dataset['train']
    
    total_samples = len(train_dataset)
    world_size = torch.distributed.get_world_size()
    
    base_samples_per_gpu = total_samples // world_size
    remaining_samples = total_samples % world_size
    
    start_idx = local_rank * base_samples_per_gpu + min(local_rank, remaining_samples)
    end_idx = start_idx + base_samples_per_gpu + (1 if local_rank < remaining_samples else 0)
    
    if local_rank == 0:
        print(f"Total samples: {total_samples}")
        print(f"World size: {world_size}")
        print(f"Base samples per GPU: {base_samples_per_gpu}")
        print(f"Remaining samples: {remaining_samples}")
        
    print(f"GPU {local_rank} processing samples from {start_idx} to {end_idx} (total: {end_idx-start_idx})")
    
    local_dataset = train_dataset.select(range(start_idx, end_idx))

    model, processor = setup_model_and_processor()

    results = []
    for idx, item in enumerate(tqdm(local_dataset, 
                                  desc=f"GPU {local_rank} processing")):
        try:
            new_caption = generate_new_caption(
                model,
                processor,
                item['image'],
                item['text']
            )
            
            results.append({
                'id': start_idx + idx,
                'original_caption': item['text'],
                'enhanced_caption': new_caption,
                'image': item['image']
            })
            
        except Exception as e:
            print(f"Error processing image {start_idx + idx} on GPU {local_rank}: {e}")
            results.append({
                'id': start_idx + idx,
                'original_caption': item['text'],
                'enhanced_caption': "Error generating caption",
                'image': item['image']
            })
    
    df = pd.DataFrame(results)
    df.to_csv(f'enhanced_captions_gpu_{local_rank}.csv', index=False)
    
    torch.distributed.barrier()
    
    if local_rank == 0:
        all_results = []
        for i in range(world_size):
            try:
                partial_df = pd.read_csv(f'enhanced_captions_gpu_{i}.csv')
                all_results.append(partial_df)
            except Exception as e:
                print(f"Error reading results from GPU {i}: {e}")
        
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            final_df = final_df.sort_values('id').reset_index(drop=True)
            final_df.to_csv('enhanced_captions_final.csv', index=False)
            
            for i in range(world_size):
                try:
                    os.remove(f'enhanced_captions_gpu_{i}.csv')
                except:
                    pass
            
            print("\nExample Results:")
            print(final_df.head(3))

    torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()