
%%writefile /content/OnPro/modules/APF.py
# fixed:  .cpu
import torch.nn as nn
import torch
import torch.nn.functional as F
from kornia.augmentation import RandomMixUpV2
import numpy as np
import itertools


class AdaptivePrototypicalFeedback(nn.Module):
    def __init__(self, buffer, mixup_base_rate, mixup_p, mixup_lower, mixup_upper, mixup_alpha,
                 class_per_task):
        super(AdaptivePrototypicalFeedback, self).__init__()
        self.buffer = buffer
        self.class_per_task = class_per_task
        self.mixup_base_rate = mixup_base_rate
        self.mixup_p = mixup_p
        self.mixup_lower = mixup_lower
        self.mixup_upper = mixup_upper
        self.mixup_alpha = mixup_alpha
        self.mixup = RandomMixUpV2(p=mixup_p, lambda_val=(mixup_lower, mixup_upper),
                                    data_keys=["input", "class"]).cuda()

    def forward(self, mem_x, mem_y, buffer_batch_size, classes_mean, task_id):
        base_rate = self.mixup_base_rate
        base_sample_num = int(buffer_batch_size * base_rate)

        # Đảm bảo indices được tạo trên cùng thiết bị với mem_x nếu mem_x ở GPU
        # Tuy nhiên, np.random.choice trả về mảng numpy, sau đó chuyển sang tensor và đẩy lên cuda.
        # Ở đây, indices được tạo trên CUDA, mem_x cũng trên CUDA, nên dòng này không phải nguyên nhân lỗi.
        indices = torch.from_numpy(np.random.choice(mem_x.size(0), base_sample_num, replace=False)).cuda()
        mem_x_base = mem_x[indices]
        mem_y_base = mem_y[indices]

        mem_x_base_mix, mem_y_base_mix = self.mixup(mem_x_base, mem_y_base)

        prob_sample_num = buffer_batch_size - base_sample_num
        if prob_sample_num != 0:
            nonZeroRows = torch.abs(classes_mean).sum(dim=1) > 0
            ZeroRows = torch.abs(classes_mean).sum(dim=1) == 0
            class_num = classes_mean.shape[0]

            # DÒNG CẦN SỬA ĐẦU TIÊN (đã được bạn đề cập trong câu hỏi trước)
            # nonZero_class = torch.arange(class_num)[nonZeroRows] # Dòng cũ
            nonZero_class = torch.arange(class_num)[nonZeroRows.cpu()] # Dòng đã sửa

            # DÒNG CẦN SỬA THỨ HAI: Zero_class
            # ZeroRows có thể ở GPU, và torch.arange(class_num) ở CPU
            # Zero_class = torch.arange(class_num)[ZeroRows] # Dòng cũ
            Zero_class = torch.arange(class_num)[ZeroRows.cpu()] # Dòng đã sửa

            classes_mean = classes_mean[nonZeroRows] # nonZeroRows ở GPU, classes_mean cũng ở GPU -> OK

            dis = torch.pdist(classes_mean)  # K*(K-1)/2

            sample_p = F.softmax(1 / dis, dim=0)

            mix_x_by_prob, mix_y_by_prob = self.make_mix_pair(sample_p, prob_sample_num, nonZero_class, Zero_class,
                                                              task_id)

            mem_x = torch.cat([mem_x_base_mix, mix_x_by_prob])
            mem_y_mix = torch.cat([mem_y_base_mix, mix_y_by_prob])

            origin_mem_y, mix_mem_y, mix_lam = mem_y_mix[:, 0], mem_y_mix[:, 1], mem_y_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
        else:
            mem_x = mem_x_base_mix
            origin_mem_y, mix_mem_y, mix_lam = mem_y_base_mix[:, 0], mem_y_base_mix[:, 1], mem_y_base_mix[:, 2]
            new_mem_y = (1 - mix_lam) * origin_mem_y + mix_lam * mix_mem_y
            mem_y = new_mem_y
            mem_y_mix = mem_y_base_mix

        return mem_x, mem_y, mem_y_mix

    def make_mix_pair(self, sample_prob, prob_sample_num, nonZero_class, Zero_class, current_task_id):
        start_i = 0
        end_i = (current_task_id + 1) * self.class_per_task

        # sample_num_per_class_pair là kết quả của phép nhân tensor (sample_prob) với số nguyên (prob_sample_num),
        # sau đó làm tròn. sample_prob ở GPU, nên sample_num_per_class_pair cũng sẽ ở GPU.
        # Khi dùng nó để làm chỉ mục cho add_idx hay reduce_idx,
        # bản thân add_idx và reduce_idx cần phải ở cùng thiết bị với sample_num_per_class_pair.
        # Tuy nhiên, torch.randperm và torch.nonzero thường tạo ra tensor trên CPU nếu không chỉ định.
        # Để đảm bảo, ta cần chuyển chúng về cùng thiết bị.

        sample_num_per_class_pair = (sample_prob * prob_sample_num).round()
        diff_num = int((prob_sample_num - sample_num_per_class_pair.sum()).item())
        if diff_num > 0:
            # DÒNG CẦN SỬA THỨ BA: add_idx
            # add_idx = torch.randperm(sample_num_per_class_pair.shape[0])[:diff_num] # Dòng cũ
            add_idx = torch.randperm(sample_num_per_class_pair.shape[0], device=sample_num_per_class_pair.device)[:diff_num] # Dòng đã sửa
            sample_num_per_class_pair[add_idx] += 1
        elif diff_num < 0:
            # DÒNG CẦN SỬA THỨ TƯ: reduce_idx
            # torch.nonzero trả về tensor trên CPU theo mặc định, cần chuyển về cùng device với sample_num_per_class_pair
            # reduce_idx = torch.nonzero(sample_num_per_class_pair, as_tuple=True)[0] # Dòng cũ
            reduce_idx = torch.nonzero(sample_num_per_class_pair, as_tuple=True)[0].to(sample_num_per_class_pair.device) # Dòng đã sửa
            # reduce_idx_ = torch.randperm(reduce_idx.shape[0])[:-diff_num] # Dòng cũ
            reduce_idx_ = torch.randperm(reduce_idx.shape[0], device=reduce_idx.device)[:-diff_num] # Dòng đã sửa
            reduce_idx = reduce_idx[reduce_idx_]
            sample_num_per_class_pair[reduce_idx] -= 1

        assert sample_num_per_class_pair.sum() == prob_sample_num

        # Các dòng dưới đây (x_indices, y_indices, y = self.buffer.y, v.v.)
        # Các tensor này được lấy từ self.buffer, thường là CPU.
        # Nếu buffer.x và buffer.y là trên CPU, thì các chỉ mục này cũng sẽ ở CPU.
        # Vấn đề thường xảy ra khi tensor dùng làm chỉ mục (ví dụ nonZeroRows) ở GPU
        # còn tensor bị lập chỉ mục (ví dụ torch.arange) ở CPU.
        # Trong trường hợp này, x_indices và y_indices sẽ được tạo trên CPU, và y cũng sẽ được chuyển về CPU (nếu buffer.y ở đó).
        # Không cần .cpu() ở đây vì chúng đã cùng thiết bị (CPU) hoặc sẽ được xử lý khi lấy dữ liệu từ buffer.
        x_indices = torch.arange(self.buffer.x.shape[0])
        y_indices = torch.arange(self.buffer.y.shape[0])
        y = self.buffer.y
        _, y = torch.max(y, dim=1) # y ở đây sẽ là kết quả của torch.max, sẽ trên cùng thiết bị với input y.

        class_x_list = []
        class_y_list = []
        class_id_map = {}
        for task_id in range(start_i, end_i):
            if task_id in Zero_class: # Zero_class đã được đưa về CPU ở trên
                continue
            indices = (y == task_id) # y và task_id (là int) cần cùng thiết bị. y ở đây có thể là GPU.
                                     # Nếu y là GPU, indices sẽ là GPU.
                                     # -> x_indices[indices] sẽ hoạt động bình thường nếu x_indices ở GPU.
                                     # Tuy nhiên, x_indices được tạo từ torch.arange(self.buffer.x.shape[0]) -> CPU.
                                     # Đây có thể là một điểm lỗi tiềm năng nếu y luôn ở GPU.
                                     # Cần đảm bảo y và x_indices cùng thiết bị.
                                     # Cách an toàn nhất là chuyển y về CPU trước khi so sánh với task_id nếu x_indices luôn CPU.
                                     # Hoặc chuyển x_indices về GPU nếu y luôn GPU.
                                     # Để đơn giản và khắc phục lỗi hiện tại:

            # DÒNG CẦN SỬA THỨ NĂM (và thứ SÁU): Đảm bảo y và indices cùng thiết bị.
            # Vì x_indices và y_indices thường là CPU (tạo từ arange), ta sẽ chuyển y về CPU để so sánh.
            indices = (y.cpu() == task_id) # Sửa y.cpu()

            if not any(indices):
                continue

            # Khi truy cập x_indices[indices] và y_indices[indices], x_indices/y_indices (CPU)
            # và indices (GPU nếu không sửa y.cpu()) sẽ gây lỗi.
            # Với sửa đổi trên (y.cpu()), indices cũng sẽ ở CPU. Vậy nên không cần thêm .cpu() ở đây.
            class_x_list.append(x_indices[indices])
            class_y_list.append(y_indices[indices])
            class_id_map[task_id] = len(class_y_list) - 1

        mix_images = []
        mix_labels = []

        for idx, class_pair in enumerate(itertools.combinations(nonZero_class.tolist(), 2)):
            n = int(sample_num_per_class_pair[idx].item())
            if n == 0:
                continue
            first_class_y = class_pair[0]
            second_class_y = class_pair[1]

            if first_class_y not in class_id_map:
                first_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                first_class_y = int(first_class_y)
            if second_class_y not in class_id_map:
                second_class_y = np.random.choice(list(class_id_map.keys()), 1)[0]
                second_class_y = int(second_class_y)

            first_class_idx = class_id_map[first_class_y]
            second_class_idx = class_id_map[second_class_y]

            # DÒNG CẦN SỬA THỨ BẢY và TÁM: first_class_sample_idx và second_class_sample_idx
            # np.random.choice trả về mảng numpy (CPU). Khi chuyển thành torch tensor, nó sẽ ở CPU.
            # Nếu class_x_list[...].tolist() là từ GPU, việc chuyển sang list, rồi numpy, rồi torch lại là hơi vòng vèo.
            # Tuy nhiên, nếu buffer.x ở GPU thì class_x_list[first_class_idx] sẽ chứa các chỉ mục trên CPU.
            # Và first_class_sample_idx cũng sẽ ở CPU.
            # Khi dùng first_class_x = self.buffer.x[first_class_sample_idx], nếu buffer.x ở GPU
            # và first_class_sample_idx ở CPU, sẽ gây lỗi.
            # Vậy, cần đẩy first_class_sample_idx lên GPU.

            # first_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[first_class_idx].tolist(), n)).long() # Dòng cũ
            first_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[first_class_idx].tolist(), n)).long().cuda() # Dòng đã sửa

            # second_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[second_class_idx].tolist(), n)).long() # Dòng cũ
            second_class_sample_idx = torch.from_numpy(np.random.choice(class_x_list[second_class_idx].tolist(), n)).long().cuda() # Dòng đã sửa

            first_class_x = self.buffer.x[first_class_sample_idx]
            second_class_x = self.buffer.x[second_class_sample_idx]

            mix_pair, mix_lam = self.mixup_by_input_pair(first_class_x, second_class_x, n)
            mix_y = torch.zeros(n, 3)
            mix_y[:, 0] = first_class_y
            mix_y[:, 1] = second_class_y
            mix_y[:, 2] = mix_lam

            mix_images.append(mix_pair)
            mix_labels.append(mix_y)

        mix_images_by_prob = torch.cat(mix_images).cuda()
        mix_labels_by_prob = torch.cat(mix_labels).cuda()

        return mix_images_by_prob, mix_labels_by_prob

    def mixup_by_input_pair(self, x1, x2, n):
        if torch.rand([]) <= self.mixup_p:
            # lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, n)).cuda() # Dòng cũ
            # lam_ = lam.unsqueeze(0).unsqueeze(0).unsqueeze(0).view(-1, 1, 1, 1) # Dòng cũ

            # DÒNG CẦN SỬA THỨ CHÍN và MƯỜI: lam và lam_
            # lam cần ở cùng thiết bị với x1 (thường là GPU) để phép toán sau đó không lỗi.
            lam = torch.from_numpy(np.random.beta(self.mixup_alpha, self.mixup_alpha, n)).to(x1.device) # Dòng đã sửa
            lam_ = lam.unsqueeze(0).unsqueeze(0).unsqueeze(0).view(-1, 1, 1, 1) # Dòng này sẽ giữ cùng device với lam
        else:
            lam = 0
            lam_ = 0

        # DÒNG CẦN SỬA THỨ MƯỜI MỘT và MƯỜI HAI: lam và lam_ (trong trường hợp else)
        # lam và lam_ cần ở cùng thiết bị với x1 để phép toán (1 - lam_) * x1 + lam_ * x2 không lỗi.
        # torch.tensor(..., dtype=x1.dtype) sẽ tạo tensor trên CPU nếu không có device.
        # lam = torch.tensor(lam, dtype=x1.dtype) # Dòng cũ
        lam = torch.tensor(lam, dtype=x1.dtype, device=x1.device) # Dòng đã sửa
        # lam_ = torch.tensor(lam_, dtype=x1.dtype) # Dòng cũ
        lam_ = torch.tensor(lam_, dtype=x1.dtype, device=x1.device) # Dòng đã sửa

        image = (1 - lam_) * x1 + lam_ * x2
        return image, lam