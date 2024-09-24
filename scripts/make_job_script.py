def make_yaml(params):
    yaml_content = f"""
apiVersion: batch/v1
kind: Job
metadata:
 generateName: ks-sac-
 labels:
   kueue.x-k8s.io/queue-name: eidf079ns-user-queue
spec:
 completions: 1
 parallelism: 1
 template:
  spec:
   restartPolicy: Never
   containers:
   - name: pytorch-con
     image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
     volumeMounts:
      - mountPath: /dev/shm
        name: dshm
     command: [/bin/bash, -c, --]
     args: 
       - |
         export OMP_NUM_THREADS=1;
         mkdir build;
         cd build;
         apt update; apt upgrade -y; apt install git vim -y;
         git clone https://github.com/maxweissenbacher/basic_sac_ks.git;
         cd basic_sac_ks;
         git checkout main;
         pip install -r requirements_ks.txt;
         wandb login ac8ec66b318e6624089b2723c3174f01a850416c;
         python main_sac.py env.nu=0.04 logger.project_name=SAC_KS_gridsearch_UTDratio env.exp_name=KS_utd_ratio_{params["utd"]} optim.utd_ratio={params["utd"]} collector.frames_per_batch=500;
     resources:
      requests:
       cpu: 8
       memory: "64Gi"
      limits:
       cpu: 8
       memory: "64Gi"
       nvidia.com/gpu: 1
   nodeSelector:
    nvidia.com/gpu.product: NVIDIA-A100-SXM4-80GB
   volumes:
   - name: dshm
     emptyDir:
      medium: Memory

"""
    return yaml_content

if __name__ == '__main__':
    import numpy as np

    def gamma_to_horizon(x):
        return int(1/(1-x))
    
    def horizon_to_gamma(x):
        return 1-1/x

    # horizon_values = [2, 5, 10, 20, 50, 100, 200, 500]
    # frames_per_batch = [50, 100, 200, 500, 1000, 5000]
    utd_ratios = [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 5.0]

    counter = 0
    for j, v in enumerate(utd_ratios):
        params = {}
        # params['gamma'] = horizon_to_gamma(v)
        # params['fpb'] = v
        params['utd'] = v

        with open(f'utd_ratio_scripts/job_{counter}.yaml', 'w') as file:
            file.write(make_yaml(params))

        counter += 1
