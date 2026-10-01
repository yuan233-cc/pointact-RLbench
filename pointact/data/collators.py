import torch
from transformers.data.data_collator import DefaultDataCollator

from pointact.constants import IGNORE_INDEX


def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


class DataCollator(DefaultDataCollator):
    """Collate multimodal examples."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_pixel_video_values = []
        batch_video_thw = []
        batch_image_thw = []
        batch_second_per_grid_ts = []
        batch_points, npoints_in_batch = [], []

        batch_actions = []
        batch_states = []
        batch_action_is_pad = []

        is_labels_provided = "labels" in examples[0]
        for example in examples:
            keys = example.keys()
            batch_input_ids.append(example["input_ids"])

            if is_labels_provided:
                batch_label_ids.append(example["labels"])

            if "pixel_values_videos" in keys:
                batch_pixel_video_values.append(example["pixel_values_videos"])
                batch_video_thw.append(example["video_grid_thw"])
            elif "pixel_values" in keys:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

            if "second_per_grid_ts" in keys:
                batch_second_per_grid_ts.extend(example["second_per_grid_ts"])

            if "actions" in keys:
                batch_actions.append(example["actions"])
                batch_action_is_pad.append(example["action_is_pad"])

            if "states" in keys:
                batch_states.append(example["states"])

            if "points" in keys:
                batch_points.append(example["points"])
                npoints_in_batch.append(example["npoints_in_batch"])

        input_id_lens = torch.LongTensor([len(input_ids) for input_ids in batch_input_ids])
        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = input_ids != self.pad_token_id
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_id_lens": input_id_lens,
        }

        if is_labels_provided:
            labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)
            data_dict["labels"] = labels

        if len(batch_pixel_values) > 0:
            data_dict["pixel_values"] = torch.cat(batch_pixel_values, dim=0)
            data_dict["image_grid_thw"] = torch.cat(batch_image_thw, dim=0)

        if len(batch_pixel_video_values) > 0:
            data_dict["pixel_values_videos"] = torch.cat(batch_pixel_video_values, dim=0)
            data_dict["video_grid_thw"] = torch.cat(batch_video_thw, dim=0)

        if len(batch_second_per_grid_ts) > 0:
            data_dict["second_per_grid_ts"] = batch_second_per_grid_ts

        if len(batch_actions) > 0:
            data_dict["actions"] = torch.cat(batch_actions, dim=0)
            data_dict["action_is_pad"] = torch.cat(batch_action_is_pad, dim=0)

        if len(batch_states) > 0:
            data_dict["states"] = torch.cat(batch_states, dim=0)

        if len(batch_points) > 0:
            data_dict["points"] = torch.cat(batch_points, dim=0)
            data_dict["npoints_in_batch"] = torch.LongTensor(npoints_in_batch)
        if "target_points" in examples[0]:
            if any("target_points" not in example or "target_input_mask" not in example for example in examples):
                raise ValueError("Cannot mix samples with and without target reconstruction labels")
            data_dict["target_points"] = pad_sequence(
                [example["target_points"] for example in examples]
            )
            data_dict["target_counts"] = torch.LongTensor(
                [len(example["target_points"]) for example in examples]
            )
            data_dict["target_input_mask"] = torch.cat(
                [example["target_input_mask"] for example in examples]
            )
        if "material_rgb" in examples[0]:
            if any("material_rgb" not in example for example in examples):
                raise ValueError("Cannot mix material-conditioned and plain examples in one batch")
            data_dict["material_rgb"] = torch.stack([example["material_rgb"] for example in examples])
            data_dict["polar_dense"] = torch.stack([example["polar_dense"] for example in examples])
            data_dict["point_pixel_indices"] = torch.cat(
                [example["point_pixel_indices"] for example in examples])
            candidate_counts = [len(example["material_candidates"]) for example in examples]
            max_candidates = max(candidate_counts)
            feature_size = examples[0]["material_candidates"].shape[-1]
            candidates = examples[0]["material_candidates"].new_zeros(
                len(examples), max_candidates, feature_size)
            mask = torch.zeros(len(examples), max_candidates, dtype=torch.bool)
            for index, example in enumerate(examples):
                count = candidate_counts[index]
                candidates[index, :count] = example["material_candidates"]
                mask[index, :count] = True
            data_dict["material_candidates"] = candidates
            data_dict["material_candidate_mask"] = mask

        has_polar = ["polar_images" in example for example in examples]
        if any(has_polar) and not all(has_polar):
            raise ValueError("Cannot mix Polar-token and baseline examples in one batch")
        if all(has_polar):
            required = ("polar_images", "polar_K", "T_camera_from_model", "view_valid")
            optional = ("T_model_from_world", "pixel_valid", "polar_pixel_transform")
            for key in required:
                if any(key not in example for example in examples):
                    raise ValueError(f"Polar-token examples require {key}")
                data_dict[key] = torch.stack([example[key] for example in examples])
            for key in optional:
                present = [key in example for example in examples]
                if any(present) and not all(present):
                    raise ValueError(f"Cannot mix examples with and without {key}")
                if all(present):
                    data_dict[key] = torch.stack([example[key] for example in examples])
            depth_keys = ("observed_depth", "observed_depth_valid")
            depth_present = [all(key in example for key in depth_keys) for example in examples]
            if any(depth_present) and not all(depth_present):
                raise ValueError("Cannot mix examples with and without sparse observed depth")
            if all(depth_present):
                for key in depth_keys:
                    data_dict[key] = torch.stack([example[key] for example in examples])

        return data_dict
