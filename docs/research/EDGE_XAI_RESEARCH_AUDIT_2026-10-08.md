# Đánh giá dự án và kế hoạch nghiên cứu Edge + XAI trong 6–8 tháng

Ngày rà soát và cập nhật kế hoạch: 08/10/2026. Lộ trình 6–8 tháng này thay thế kế hoạch 12 tuần trước đó. Mục tiêu: hoàn thiện nghiên cứu và bản thảo để nộp tạp chí phù hợp, hướng tới Q1/Q2; dự kiến nộp tháng 7–8 của dự án, khoảng tháng 05–06/2027 nếu bắt đầu tháng 10/2026. Không cam kết được chấp nhận/xuất bản trong thời hạn này. Các quan sát mã nguồn và số đo bên dưới vẫn thuộc đợt audit ngày 08/10/2026, không phải kết quả thí nghiệm mới.

## 1. Kết luận và giới hạn khảo sát

Hướng đề xuất: **nhận dạng cử chỉ bằng model nhỏ, giải thích bằng khái niệm theo thời gian, cho phép con người sửa bằng chứng và tính lại dự đoán**. Ưu tiên chứng minh chất lượng can thiệp khi keypoint bị thiếu/sai và khi triển khai model nhẹ. Không lấy việc có KG, attention, ba nhánh logits hay thêm giao diện giải thích làm đóng góp mới riêng lẻ.

Đã rà các nhóm chức năng trong mã nguồn: extraction; chuẩn hóa/cache; manifest/cắt/audit IPN; predicate/event/instance graph/template/matcher; neural baselines; các phiên bản fusion V2–V5 và train/test; demo/explanation; benchmark và script tái lập. Đã đọc artifact đánh giá và benchmark hiện có. Đây là kiểm toán kiến trúc và nghiên cứu, không phải xác nhận mọi đường chạy hoặc kiểm tra trực quan toàn bộ video; dependencies, binary checkpoint và toàn bộ dữ liệu không được đọc như mã nguồn. Review độc lập dùng để đối chiếu các nhánh model. Skill architect-review định hướng đánh giá cả hệ thống, thay vì chỉ classifier.

Đã chạy forward CPU với 8 cấu hình baseline sẵn có; chưa huấn luyện model mới, chưa đo accuracy mới, chưa đo thiết bị biên. Tìm kiếm tài liệu có mục tiêu đến ngày khảo sát, không phải systematic review đầy đủ. Những khoảng trống dưới đây là giả thuyết cần xác minh thêm bằng đọc full text/code và thí nghiệm.

## 2. Nền tảng hiện có và vấn đề cần xử lý trước bài báo

| Quan sát có bằng chứng | Ý nghĩa | Hành động ưu tiên |
|---|---|---|
| RTMW-L + detector trong artifact có khoảng 82,02M tham số; RTMW-M + detector khoảng 56,92M | Extractor là phần nặng đáng kể; giảm classifier chưa đủ | Đo từng công đoạn và thay extractor trong thử nghiệm có kiểm soát |
| Artifact CPU RTMW-M/L: median khoảng 1.021/1.437 ms mỗi frame; chỉ 27 clip, 216 frame lấy mẫu | Đây là số đo cũ trên CPU với protocol nhỏ, không phải benchmark edge hay đầy đủ video | Tái lập batch=1, video mode/tracking, p50/p95, throughput và năng lượng trên máy đích |
| Báo cáo best_test.pt ghi rõ chọn checkpoint bằng test; final 96,18%, neural 95,28%, KG-guided 95,93% trên 1.229 mẫu | Kết quả exploratory, không phải locked holdout; chênh lệch không chứng minh tác động nhân quả của KG | Dùng validation chọn checkpoint; nếu test đã dùng để phát triển, cần holdout mới hoặc protocol ngoài độc lập |
| kg_logits V4 nhận cả neural_memory và graph_memory | Tên phù hợp là template/KG-guided head, không phải graph-only | Giữ ba đầu ra để phân tích nội bộ; thêm neural baseline train độc lập và control matched-capacity |
| rule matcher không đi vào final logits | Lời giải thích matcher có thể khác lý do classifier | Ghi rõ audit signal; không mô tả nó như bằng chứng nhân quả của final |
| requirement_evidence: satisfied lấy từ membership trong graph; attention graph là compressor slots | Có requirement được quan sát không đồng nghĩa nó quyết định dự đoán | Can thiệp có kiểm soát và đo delta logits; ánh xạ temporal evidence đúng frame gốc |
| Chuẩn hóa phụ thuộc hai vai; frame invalid đặt xy=0 nhưng vẫn giữ score | Hand-only extractor không thể thay trực tiếp; mask/unknown cần rõ ràng | Thiết kế representation bàn tay cục bộ + quỹ đạo toàn cục, validity mask và quality riêng |
| Tensor 64 frame dùng lấy mẫu đều hoặc lặp frame cuối; không có padding mask trong pooling | Mất thời gian thực/độ dài, có thể lệch số lần click | Giữ timestamp, mask, valid length; kiểm thử frame-rate và tốc độ động tác |
| Predicate window=5, threshold=0.08; cache fallback đánh valid=True cho mọi frame | Semantic thay đổi theo FPS, resampling và nhánh dữ liệu | Khóa một preprocessing contract, cửa sổ theo thời gian thực; version/hash cache |
| Demo dùng event builder cũ; có pipeline phase_v2 riêng | Train/deploy có thể lệch nếu chọn graph cache khác nhau | Lưu phiên bản builder trong config/checkpoint; kiểm tra parity |
| Template buffers không persistent; vocabulary dựng lại từ thư mục ngoài | Đổi template/concept order có thể đổi semantics khi nạp weight | Bundle template, vocabulary, label map, preprocessing và hashes cùng checkpoint |
| Mặc định reverse cả cạnh BEFORE/AFTER; concept node dùng chung giữa nhiều event | Hướng temporal của adjacency bị đối xứng; requirement/fact theo node có thể bị ghi đè nếu nhiều event/subject cùng dùng concept | Giữ directed temporal edges và biểu diễn requirement theo event–subject–concept; kiểm thử lặp/cặp đảo thứ tự |
| V5 là subclass rỗng của V4; khác template | Không có kiến trúc V5 mới trong code này | Gọi đúng là template tuning; lưu nguồn và dữ liệu đã dùng để chỉnh |
| Neural dataset có fallback glob toàn cache và lấy kết quả đầu tiên | Trùng filename có nguy cơ lấy nhầm split; chưa chứng minh đã xảy ra | Resolve theo manifest duy nhất; fail nếu thiếu/trùng, hash tensor dùng cho mỗi mẫu |
| IPN hiện loại subject 1CM42_3: 116/28/52 video, 2.443/589/1.101 clip | Training protocol đã khác bộ đầy đủ; test vẫn có 52 video | Công bố lý do loại, audit nguồn và dùng cùng exclusion cho mọi baseline |
| IPN B0A/B0B nhiều mẫu hơn các lớp động; D0X không trong manifest | Accuracy có thể che lỗi lớp hiếm; chưa chứng minh continuous recognition | Macro-F1/per-class recall; continuous/D0X là extension riêng |
| Script benchmark gọi benchmark_keypoint_models.py nhưng file không có trong tracked inventory; requirements chưa pin | Khó tái lập từ clone sạch | Đóng gói script cần thiết, lock môi trường, seed/config/hash và kiểm tra clone sạch |

Bằng chứng mã/artifact:

- `final/code/src_neurosymbolic/models/cross_attention_v4.py:177–237`: hai memory, scorer, fusion.
- `final/code/src_neurosymbolic/train_cross_attention_v4.py:508–574`: lựa chọn và ghi best_test.
- `results/experiments/legacy/final_outputs/evidence/model_evaluation_evidence/test_selected_checkpoint/summary.json`: warning và metric cũ.
- `results/experiments/legacy/final_outputs/evidence/keypoint_benchmark_evidence/{combined_summary,run_metadata,rtmw_l_run}.json`: protocol/timing extractor cũ. Có trường mean_valid_keypoint_confidence > 1 ở L; phải audit định nghĩa/thang score trước khi diễn giải như xác suất.
- `final/code/src_neurosymbolic/demo_video_v4.py:225–294,388–421`: evidence/attention/matcher.
- `final/code/src_neurosymbolic/demo_ui/app.py:357–402`: matcher là audit signal, không override logits.
- `final/code/src_neurosymbolic/pose/normalize_keypoints.py:48–78` và `models/keypoint_dataset.py:143–169`: validity/resampling.
- `final/code/src_neurosymbolic/precompute_instance_graphs_phase_v2.py:22–30,99–117`: raw/cache fallback và threshold.
- `final/code/src_neurosymbolic/models/cross_attention_v2.py:175–192`: nonpersistent template buffers.
- `final/code/src_neurosymbolic/models/cross_attention_v2.py:204–207,237–248,283–295` và `models/cross_attention_v4.py:72–89`: temporal adjacency và requirement overwrite có điều kiện.
- `final/code/src_neurosymbolic/models/cross_attention_v5.py` và `final/code/neural_only_experiments/dataset.py:42–73`: version V5/cache fallback.
- `external_data/ipn_processed/split_subjects.json`, `scripts/create_ipn_manifest.py`.

Không cần huấn luyện KG-only riêng để hoàn thành nghiên cứu này. Tuy nhiên, so sánh ba head trong một checkpoint chỉ đo hành vi của mô hình đồng huấn luyện. Để nói tri thức giúp, cần control neural độc lập, kiến trúc/capacity/training budget phù hợp, template đúng so với hoán đổi/ngẫu nhiên và phép tắt/bỏ graph được mô tả rõ.

Chi phí fusion có thể giảm bằng pre-tensorize graph thay JSON parse trong forward, cache template embedding khi model eval/weight đã cố định, giới hạn graph/event theo cửa sổ và tránh expand dư theo mọi class. Phải invalidate cache khi sửa template/weight. Đây là tối ưu kỹ thuật, không mặc nhiên là đóng góp bài báo.

## 3. Thử model nhẹ: đã đo gì, nên thử gì tiếp

Lệnh tái lập:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_edge_candidates.py --output docs\research\edge_candidates_cpu_smoke.json
```

Điều kiện: CPU Windows hiện tại, PyTorch 2.14.0+cpu, một intra-op thread, FP32, batch=1, input [1,64,52,3], 13 lớp, trọng số ngẫu nhiên, 20 warmup + 100 lượt/model. File JSON lưu CPU/platform. Chưa kiểm soát affinity, thứ tự model hoặc nhiều process; p95 nhiễu. Không so trực tiếp với benchmark extractor cũ như cùng protocol.

| Backbone | Width/layers | Tham số | Median forward ms | p95 ms |
|---|---:|---:|---:|---:|
| TCN có sẵn | 128/3 | 236.429 | 2,074 | 4,856 |
| BiGRU | 128/3 | 262.541 | 12,946 | 25,133 |
| Transformer | 128/3 | 633.869 | 3,685 | 9,072 |
| MLP-Mixer | 128/3 | 485.453 | 2,706 | 8,031 |
| TCN có sẵn | 64/2 | 48.461 | 0,721 | 2,597 |
| BiGRU | 64/2 | 53.069 | 8,070 | 13,886 |
| Transformer | 64/2 | 115.405 | 1,529 | 3,882 |
| MLP-Mixer | 64/2 | 115.277 | 1,116 | 3,909 |

Ưu tiên train TCN-64/2 và Transformer-64/2 trước. Sau đó thêm depthwise-separable TCN nhỏ và một lightweight skeleton GCN/HAN phù hợp, không mở cuộc đua 10 kiến trúc. TCN hiện dùng symmetric padding, BiGRU dùng hai chiều: chưa được gọi là causal streaming. Thay width và depth đồng thời ở bảng này là lựa chọn cấu hình, không phải ablation tách tác động từng yếu tố.

Extractor: thử MediaPipe Hand Landmarker ở VIDEO/LIVE_STREAM với 1–2 bàn tay, cân nhắc Pose Lite khi cần vai/thân. MediaPipe trả 21 landmarks mỗi bàn tay và có tracking tránh detector mỗi frame; không cung cấp cùng schema/score với RTMW. Cần giữ handedness, root trajectory, hand scale, timestamp, mask và định nghĩa confidence mới; không điền vai bằng zero rồi dùng chuẩn hóa cũ. Tài liệu Google có benchmark Pixel 6 nhưng không được dùng thay đo trên máy đích. [Tài liệu chính thức](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

INT8 là thử nghiệm sau khi FP32 chạy ổn; đo parity logits, accuracy, concept F1 và hiệu quả can thiệp. Số tham số/FLOPs nhỏ không bảo đảm latency nhỏ: ngay smoke BiGRU ít tham số nhưng chậm hơn TCN. Chốt runtime theo thiết bị; chưa mặc định Android/Jetson/Raspberry Pi khi chưa có thông tin phần cứng.

## 4. Related work và những gì không còn mới

| Công trình | Đã làm | Hệ quả cho đề tài |
|---|---|---|
| [Lightweight hand GCN, 2021](https://arxiv.org/abs/2104.04255) | GCN nhẹ và học connectivity cho gesture | Nhỏ hóa GCN không tự tạo novelty |
| [HAN, Pattern Recognition 2025](https://doi.org/10.1016/j.patcog.2025.111343) | Hierarchical self-attention hiệu quả cho skeleton gesture | Cần baseline hiệu quả gần đây, không chỉ Transformer lớn |
| [Part-level KD, Signal Processing 2024](https://arxiv.org/abs/2404.18206) | Teacher skeleton tốt hướng dẫn student skeleton chất lượng thấp | Robustness + distillation chung đã có |
| [CBM, ICML 2020](https://proceedings.mlr.press/v119/koh20a.html) | Dự đoán concept trước class, sửa concept rồi suy luận lại | Human concept editing là nền tảng đã có |
| [Deep Concept Reasoner, ICML 2023](https://proceedings.mlr.press/v202/barbiero23a.html) | Logic trên concept để tạo quyết định diễn giải được | Rule reasoning không phải đóng góp mới độc lập |
| [PCBEAR, CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/XAI4CV/html/Lee_PCBEAR_Pose_Concept_Bottleneck_for_Explainable_Action_Recognition_CVPRW_2025_paper.html) | Pose concepts tĩnh/động, action recognition và can thiệp | Không claim đầu tiên đưa temporal concepts vào action XAI |
| [REASON, IJCAI 2026](https://www.ijcai.org/proceedings/2026/138) | Skeleton encoder, concept decoder không gian/thời gian, differentiable logic | Đối thủ gần nhất của tuyên bố skeleton + neuro-symbolic + XAI |
| [CB2M, ICML 2024](https://proceedings.mlr.press/v235/steinmann24a.html) | Ghi nhớ/tái áp dụng can thiệp, hỏi sửa concept | Lưu feedback người dùng rồi dùng lại cũng đã có |
| [FIGS-BD, preprint 2025](https://arxiv.org/abs/2503.06730) | Distill concept-to-target thành cây và hướng dẫn can thiệp | Distillation + intervention không tự mới |
| [Concept leakage, preprint 2025](https://arxiv.org/abs/2506.04877) | Khái niệm dự đoán đúng vẫn có thể mang thông tin ngoài semantics | Cần kiểm tra soft concept/embedding leakage |
| [Human–AI CBM study, preprint 2026](https://arxiv.org/abs/2608.25581) | Lợi ích can thiệp phụ thuộc task, khả năng nhận biết concept và tương tác | Không suy ra con người hiểu/hữu ích chỉ từ oracle experiment |

REASON đọc được bản PDF; một số publisher chặn full text, nên HAN và một số ứng viên mới chỉ được đối chiếu ở abstract/metadata. Có tín hiệu gần về MAG-KD cho sign-language edge qua DOI 10.1109/ACCESS.2026.3681176; cần đọc toàn văn trước khi chốt hướng distillation. Không xem ít kết quả tìm kiếm là bằng chứng chưa ai làm.

## 5. Đề tài đề xuất: khái niệm thời gian có thể sửa trên thiết bị biên

Tên làm việc: **Intervenable Temporal Concepts for Lightweight Dynamic Hand Gesture Recognition**. Đây là tên định hướng, không phải khẳng định novelty đã được xác lập.

Câu hỏi trung tâm: khi chuyển sang pose extractor/model nhẹ, liệu biểu diễn khái niệm có gắn đoạn thời gian và trạng thái chưa quan sát được có giúp con người sửa lỗi hiệu quả, với ít thao tác, mà vẫn đáp ứng ngân sách thiết bị?

Pipeline đề xuất:

```text
Camera -> hand/pose tracker nhẹ -> landmarks + mask + timestamp
       -> TCN nhỏ -> concept theo đoạn thời gian
       -> quan hệ thứ tự/số lần + head có quy tắc/đóng góp tường minh -> nhãn
                               ^
                     người dùng sửa concept/đoạn
```

Concept ban đầu khoảng 12–20 nhóm, xác nhận từ train: cấu hình ngón, trạng thái mở/khép, hướng quỹ đạo, co/duỗi ngón, số chu kỳ, thứ tự pha. Dùng trạng thái `observed/unknown` riêng, không ép tay bị che thành “không có động tác”. Không dùng nhãn class để tự sinh ground-truth concept cho từng clip.

Giữ vocab nhỏ và có thể quan sát bằng mắt. Nhóm direction/count dùng categorical bins; ràng buộc các lựa chọn loại trừ nhau. Giữ cả local hand shape và global motion: root-normalization hoàn toàn có thể xóa hướng di chuyển cần phân biệt. Các concept dự đoán phải có thời điểm/bằng chứng video; thời gian chỉ để hiển thị mà không vào reasoning thì không được claim temporal reasoning.

MVP dùng sequential concept learning: học concept từ annotation/weak label đã kiểm tra; đóng băng predictor, train sparse additive head trên concept dự đoán. Một số tương tác có nghĩa như “co rồi duỗi, lặp hai lần” trở thành feature/rule tường minh. Hiển thị tổng đóng góp theo logit (và bias); không gọi đó là tỷ lệ phần trăm đóng góp xác suất. Sau khi sửa concept phải tính lại các quan hệ/rule liên quan và toàn bộ class scores.

Để claim decision hoàn toàn qua concept, nhánh chính không có hidden residual từ backbone sang nhãn. Nếu residual giúp accuracy, báo riêng như hybrid và đo phần ảnh hưởng không giải thích được. Giữ V4 ba đầu ra như teacher/baseline, không bắt buộc bỏ kiến trúc hiện tại. Distillation chỉ thêm khi baseline concept đã tốt; distill logits đơn thuần có thể giữ prediction nhưng mất semantics/can thiệp.

Ba giả thuyết kiểm chứng, không phải ba module bắt buộc:

1. Can thiệp cục bộ theo đoạn thời gian sửa được nhiều lỗi hơn concept toàn clip ở cùng số thao tác, nhất là single/double click và chuyển động đảo chiều.
2. Phân biệt unknown và absent giảm “sửa hại” khi occlusion/pose dropout, so với hard-zero và confidence gating đơn giản.
3. FP32 -> INT8/extractor nhẹ có thể giữ accuracy nhưng làm đổi lời giải thích/can thiệp; đánh giá và tối ưu sự ổn định này đem lại lợi ích thực tế.

Phần có triển vọng thành đóng góp là **cơ chế + bằng chứng về can thiệp theo thời gian dưới giới hạn quan sát/tài nguyên**, không phải chỉ tích hợp các kỹ thuật đã có. Đến cuối tháng 4, nếu chưa thấy tín hiệu cải thiện đáng tin cậy trên validation so với temporal CBM đơn giản ở cùng chi phí can thiệp, phải thu hẹp/đổi giả thuyết thay vì thêm nhiều module.

Phạm vi phương pháp ưu tiên gồm ba phần gắn kết: concept theo interval có trạng thái unknown; lựa chọn đoạn/concept cần người dùng xác nhận theo ngân sách; reasoning tường minh tính lại sau sửa. Policy lựa chọn bắt đầu từ uncertainty baseline; chỉ đề xuất policy mới khi có lý do và ablation chứng minh giá trị. Không đồng thời đặt mục tiêu sáng tạo backbone, extractor, distillation, online learning và policy mới. INT8/đổi extractor trước hết là phép kiểm tra độ bền của đóng góp XAI.

## 6. Con người sửa gì, lời giải thích phải thể hiện gì

Ví dụ minh họa, chưa phải output model đã chạy: model dự đoán single click vì tìm được một chu kỳ co–duỗi ngón trỏ; đoạn thứ hai bị gán unknown. Người dùng xem đoạn video, đánh dấu chu kỳ thứ hai. Model cập nhật count/order và tính lại single/double-click scores; có thể đổi sang double click hoặc vẫn giữ nhãn nếu bằng chứng khác mạnh hơn. Không được ép nhãn đổi chỉ để demo trông thuyết phục.

UI tối thiểu: nhãn và độ tin cậy đã hiệu chỉnh; 2–4 bằng chứng ủng hộ; bằng chứng thiếu/mâu thuẫn; timeline/keypoint overlay; sửa concept trên một interval; xem trước/sau và hoàn tác. Khi không đủ hình ảnh, người dùng được chọn “không biết”. Giải thích văn bản dựng từ đúng trace tính toán; không cần LLM chạy trên edge.

Ba mức feedback phải tách:

- Sửa quan sát cho clip hiện tại: tính lại, không đổi weight/global rule.
- Sửa rule/template chung: version hóa, kiểm tra mâu thuẫn, chạy regression trên validation trước khi dùng rộng.
- Học từ feedback cho clip tương lai: hướng mở rộng, cần protocol held-out riêng; không fine-tune bằng test labels rồi báo test như cũ.

Can thiệp nội bộ model chứng minh functional dependence, không chứng minh nhân quả vật lý của cử chỉ. Attention đẹp hay văn bản hợp lý không thay được thí nghiệm này.

## 7. Bộ thí nghiệm mục tiêu cho 6–8 tháng

**Dữ liệu:** IPN isolated 13 lớp làm chính. Khóa manifest, subject split, exclusion, preprocessing; xác thực grouping subject với metadata. IPN vốn hỗ trợ isolated và continuous, nhưng pipeline hiện chỉ giải quyết isolated. [IPN gốc](https://arxiv.org/abs/2005.02134). Kiểm tra [IPN HandS 2025](https://www.mdpi.com/2076-3417/15/11/6321) để tránh làm lại annotation skeleton đã công bố; nó cùng nguồn IPN, không tính là dataset độc lập thứ hai.

Dataset thứ hai chọn trong tháng 1: NVGesture RGB để kiểm tra pipeline video/extractor; nếu tài nguyên không đủ, một benchmark skeleton như SHREC chỉ kiểm tra classifier/concept reasoning, không chứng minh RGB-to-edge. Mục tiêu là hai benchmark công khai độc lập, công bố concept nào dùng chung/concept nào đặc thù và ngân sách annotation cho từng bộ. Không dùng test label mapping khác nhau để claim zero-shot cross-dataset. [NVGesture từ NVIDIA](https://research.nvidia.com/publication/2016-06_online-detection-and-classification-dynamic-hand-gestures-recurrent-3d).

Nếu nguồn lực cho phép, thêm tập thực địa nhỏ với camera, người dùng, ánh sáng và che khuất khác để kiểm tra domain shift; khóa protocol và sự đồng ý trước thu thập. Tập này bổ sung bằng chứng ứng dụng, không thay thế benchmark công khai và không mặc nhiên là đóng góp dataset mới.

**Concept annotation:** pilot train 10–15 clip/lớp để xác định ambiguity và thời gian gán; 400–600 clip là ngân sách khởi điểm để lập kế hoạch, không phải cỡ mẫu đủ mặc định cho hai benchmark. Cuối tháng 2 chốt quy mô dựa trên độ phủ concept/lớp/subject, tỷ lệ unknown, precision mong muốn và nguồn lực. Hai người gán độc lập một phần (ví dụ 20%), đo agreement cả giá trị concept lẫn interval, phân xử. Gán frame interval, concept, unknown và bằng chứng; annotator không xem model prediction/ground-truth class nếu có thể. Weak labels từ predicate dùng để mở rộng train, không làm gold test. Dành tập concept gold độc lập để đánh giá; cần ước lượng uncertainty theo subject và công bố hạn chế kích thước mẫu. Version hóa guideline, annotation và split; phát hành annotation/toolkit nếu giấy phép dữ liệu cho phép.

**Baselines/ablations tối thiểu:**

1. TCN và Transformer nhẹ, supervised class-only.
2. V4 teacher với neural/KG-guided/fusion outputs, checkpoint chọn val.
3. Static/clip-level CBM cùng backbone và annotation budget.
4. Temporal CBM cơ bản rồi thêm cơ chế can thiệp/unknown đề xuất, từng yếu tố một.
5. Heuristic concept + rule head đơn giản để xem học thêm có thật sự cần không.
6. Một baseline nhẹ hiện đại và một concept/reasoning baseline gần (PCBEAR/REASON hoặc adaptation công bố khác biệt), tùy khả năng tái lập code.
7. Random/shuffled templates hoặc concept assignment; concept-only vs residual; FP32 vs INT8. Dữ liệu can thiệp giống nhau giữa các model.

**Metric nhận dạng:** accuracy, macro-F1, từng lớp, confusion matrix, seed mean/std (ít nhất 3 seed cho mô hình chính), paired CI. Bootstrap theo subject/video thay vì coi clip liên quan hoàn toàn độc lập. Threshold, checkpoint và hyperparameter chỉ chọn trên train/val.

**Metric XAI/can thiệp:** concept F1 và calibration; temporal IoU/F1 khi có annotation interval; accuracy/F1 sau 0/1/2/3 lần sửa; diện tích đường hiệu quả–ngân sách; correction success và harm rate; thời gian người dùng; số lần cần abstain. So sánh random selection, uncertainty selection và policy đề xuất ở cùng cost. Báo kết quả trên toàn tập lẫn tập lỗi; không chỉ chọn vài clip đẹp.

**Faithfulness:** thay một concept/interval hợp lệ, tính lại logits với các đầu vào còn lại cố định; intervention identity không sửa gì phải giữ prediction; sửa concept không liên quan là negative control. Counterfactual phải hợp lệ về cấu hình ngón, hướng, thời gian; không chỉ xóa point thành zero gây OOD. Kiểm tra cả clip cặp/biến đổi video phù hợp và gold concepts. Đo concept/rule stability giữa FP32 và INT8, không chỉ agreement nhãn. Báo soft-vs-hard bottleneck và leakage; hard bottleneck không tự bảo đảm semantic đúng.

**Human study:** pilot 5–8 người để sửa UI/protocol; cỡ mẫu chính dựa trên variance/effect-size pilot, không mặc định pilot đủ chứng minh. So sánh video + nhãn, thêm giải thích, thêm sửa concept; counterbalance các clip tương đương để giảm học thuộc. Đo đúng/sai, thời gian và thao tác; khảo sát dễ hiểu chỉ là metric phụ. Người dùng chỉ xem video/concept, không được thấy đáp án để sửa. Oracle sửa concept gold là upper bound riêng, không báo như kết quả người thật. Quy trình người tham gia/đồng ý theo cơ sở nghiên cứu.

**Robustness:** missing joints theo đoạn, occlusion, jitter, thay FPS, handedness/left-right consistency; thử extractor khác thật. Phân biệt chỉ nhiễu skeleton với occlusion ở RGB. Cùng severity schedule train/val/test, không chỉnh corruption theo test.

**Edge:** batch=1, camera resolution, số tay/người, threads, runtime/version, power mode, cold/warm, sustained run 10–15 phút. Báo extraction, classification, explanation/recompute và tổng latency p50/p95, peak RAM, model bundle size, throughput, power/energy nếu đo được. Tách thời gian chờ quan sát động tác khỏi thời gian compute. Mục tiêu dự kiến 15–30 FPS và classifier <0,25M tham số chỉ là design targets, điều chỉnh theo thiết bị; không dùng thành kết quả.

## 8. Kế hoạch và phân công

| Tuần | Người 1: dữ liệu + thiết bị | Người 2: model nhẹ | Người 3: XAI + can thiệp | Điều kiện qua mốc |
|---|---|---|---|---|
| 1–2 | Khóa split, audit lỗi/unknown; tracker nhẹ; chuẩn bị máy đích | Tái lập TCN/Transformer/V4 theo val | Vocab, annotation pilot, đọc PCBEAR/REASON/CBM | Input contract ổn; concept phân biệt được lớp; baseline chạy |
| 3–4 | Annotation gold/weak và dataset thứ hai | Train class-only và concept baseline | Temporal head + sửa interval + test faithfulness | Oracle concept có đủ năng lực; can thiệp có lợi hơn static control |
| 5–6 | Trích dữ liệu đủ, QA, corruption có kiểm soát | Train bản đề xuất, matched-budget ablations | Hoàn thiện unknown/recompute; UI pilot | Có hiệu ứng ổn định trên val; không chỉ demo |
| 7–8 | Đo edge dài, RAM/power khi có | INT8, 3 seed, dataset thứ hai | Human study, harm rate, stability | Đo end-to-end thật; kết quả XAI định lượng |
| 9–10 | Khóa artifact/protocol | Đánh giá test một lần sau khi khóa | Phân tích failure/counterfactual và giới hạn | Hai nguồn dữ liệu phù hợp claim, CI và mọi ablation cần thiết |
| 11–12 | Kiểm tra tái lập/packaging | Bảng model/edge | Viết phương pháp/XAI | Bản thảo, figures, code/config, review nội bộ và nộp |

Không chờ baseline người 2 xong mới cho người 3 làm: hai người thống nhất interface `concept_values, observation_mask, intervals, class_logits, trace` ngay tuần đầu. Người 3 có thể dùng synthetic input và sample train để làm head/can thiệp. Mỗi người một output/config riêng; chung manifest/hash.

Các điểm dừng: tuần 2 nếu concept oracle không phân biệt đủ lớp thì sửa vocab; tuần 4 nếu can thiệp không vượt baseline thì không thêm KD/online learning; tuần 8 nếu chưa có máy đích thì giảm claim edge xuống deployment feasibility và không bịa latency. Full continuous spotting, personalization lâu dài, retrain pose estimator và LLM explanations không đưa vào MVP này.

## 9. Chọn nơi nộp và kỳ vọng thực tế

Tính phù hợp là suy luận từ scope, chưa phải bảo đảm biên tập:

- [Engineering Applications of Artificial Intelligence](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976): ưu tiên nếu có ứng dụng edge thật + cơ chế AI mới + public benchmark tái lập.
- [Expert Systems with Applications](https://shop.elsevier.com/journals/expert-systems-with-applications/0957-4174): cân nhắc khi đóng góp chính là hệ thống quyết định tương tác với giải thích và đánh giá đầy đủ.
- [Knowledge-Based Systems](https://shop.elsevier.com/journals/knowledge-based-systems/0950-7051): cân nhắc nếu reasoning/can thiệp có đóng góp phương pháp rõ hơn tích hợp kỹ thuật.

Chưa xác minh quartile JCR/SJR theo năm/ngành mà đơn vị yêu cầu, vì vậy không gán nhãn Q1/Q2 cố định cho shortlist. Chốt cơ sở xếp hạng và kiểm tra lại trước khi nộp. Không đồng nhất impact factor với quartile; không lấy thời gian first decision làm thời gian accept.

Với ba người làm đều, có GPU và thiết bị sớm, 12 tuần có thể hướng tới một submission có chất lượng. Chất lượng hiện tại chưa đủ để hứa Q1/Q2: cần xử lý test-selection, chứng minh concept intervention có ích và so với prior gần. Thành công nghiên cứu không nhất thiết là accuracy cao nhất; có thể là accuracy tương đương trong biên đã chốt, ít tính toán hơn và sửa lỗi hiệu quả hơn với chi phí người dùng thấp, được chứng minh trung thực.
