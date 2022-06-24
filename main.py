import os
import sys
import logging
import argparse
import json

os.environ['RLTRADER_BASE'] = 'D:\\dev\\rltrader'
from proj import settings
from proj import utils
from proj import data_manager


if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    ''' 입력 파라미터  '''
    parser.add_argument('--mode', choices=['train', 'test', 'update', 'predict'], default='train')
    # parser.add_argument('--ver', choices=['v1', 'v2', 'v3', 'v4'], default='v2')     # RLTrader의 버전
    parser.add_argument('--name', default=utils.get_time_str())      # 로그, 가시화 파일, 신경망 모델 등의 출력 파일을 저장할 폴더 이름
    # parser.add_argument('--stock_code', nargs='+')    # 강화학습 할 주식의 종목 코드
    parser.add_argument('--rl_method', choices=['dqn', 'pg', 'ac', 'a2c', 'a3c', 'monkey'])  # 강화학습 방식 설정
    parser.add_argument('--net', choices=['dnn', 'lstm', 'cnn', 'monkey'], default='dnn')  # 가치 신경망과 정책 신경망에서 사용할 신경망 유형
    parser.add_argument('--backend', choices=['pytorch', 'tensorflow', 'plaidml'], default='pytorch')
    parser.add_argument('--start_date', default='20200101')   # 차트 데이터 및 학습 데이터 시작 날짜
    parser.add_argument('--end_date', default='20201231')    # 차트 데이터 및 학습 데이터 끝 날짜
    parser.add_argument('--lr', type=float, default=0.0001)  # 학습 속도 
    parser.add_argument('--discount_factor', type=float, default=0.7)  # 할인율
    # parser.add_argument('--balance', type=int, default=100000000)  # 주식투자 시뮬레이션을 위한 초기자본금
    args = parser.parse_args()



    ''' 학습기 파라미터 설정 '''
    output_name = f'{args.mode}_{args.name}_{args.rl_method}_{args.net}'            # 출력 파일명
    learning = args.mode in ['train', 'update']                                     # 강화학습 유무
    reuse_models = args.mode in ['test', 'update', 'predict']                       # 신경망 모델 재사용 유무
    value_network_name = f'{args.name}_{args.rl_method}_{args.net}_value.mdl'       # 가치 신경망 모델 파일명
    policy_network_name = f'{args.name}_{args.rl_method}_{args.net}_policy.mdl'     # 정책 신경망 모델 파일명
    start_epsilon = 1 if args.mode in ['train', 'update'] else 0
    num_epoches = 1000 if args.mode in ['train', 'update'] else 1
    num_steps = 5 if args.net in ['lstm', 'cnn'] else 1                             # LSTM과 CNN에서 사용할 Step 크기



    ''' Backend 설정 '''
    os.environ['RLTRADER_BACKEND'] = args.backend
    if args.backend == 'tensorflow':
        os.environ['KERAS_BACKEND'] = 'tensorflow'
    elif args.backend == 'plaidml':
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

    # 출력 경로 생성  (\output 폴더로)
    output_path = os.path.join(settings.BASE_DIR, 'output', output_name)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    # 파라미터 기록  : 입력받은 파라미터들을 JSON 형태로 저장 해서 output 폴더에 params.json 파일로 저장
    params = json.dumps(vars(args))
    with open(os.path.join(output_path, 'params.json'), 'w') as f:
        f.write(params)

    # 모델 경로 준비 (\models 폴더로)
    # 모델 포멧은 TensorFlow는 h5, PyTorch는 pickle
    value_network_path = os.path.join(settings.BASE_DIR, 'models', value_network_name)
    policy_network_path = os.path.join(settings.BASE_DIR, 'models', policy_network_name)

    # 로그 기록 설정 (\output 폴더로)
    log_path = os.path.join(output_path, f'{output_name}.log')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(format='%(message)s')
    logger = logging.getLogger(settings.LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.info(params)
    
    # Backend 설정, 로그 설정을 먼저하고 RLTrader 모듈들을 이후에 임포트해야 함 그래야 아래 클래스에도 로그 설정이 적용됨
    from proj.learners import ReinforcementLearner, DQNLearner, \
        PolicyGradientLearner, ActorCriticLearner, A2CLearner, A3CLearner

    common_params = {}
    list_stock_code = []
    list_chart_data = []
    list_training_data = []
    list_min_trading_price = []
    list_max_trading_price = []

    for stock_code in args.stock_code:
        # 차트 데이터, 학습 데이터 준비
        chart_data, training_data = data_manager.load_data(
            args.start_date, args.end_date)

        assert len(chart_data) >= num_steps
        
        # 최소/최대 단일 매매 금액 설정
        # min_trading_price = 100000
        # max_trading_price = 10000000

        # 공통 파라미터 설정
        common_params = {'rl_method': args.rl_method,                     # 강화학습 방식
            'net': args.net, 'num_steps': num_steps, 'lr': args.lr,       # 신경망 유형, Step 크기, 학습속도
            'balance': args.balance, 'num_epoches': num_epoches,          # 초기자본금, 에포크 수
            'discount_factor': args.discount_factor, 'start_epsilon': start_epsilon,     # 할인율, 시작 입실론
            'output_path': output_path, 'reuse_models': reuse_models}     # 출력 경로, 신경망 모델 재사용 유무

        # 강화학습 시작
        learner = None
        if args.rl_method != 'a3c':
            common_params.update({'stock_code': stock_code,
                'chart_data': chart_data, 
                'training_data': training_data,
                'min_trading_price': min_trading_price, 
                'max_trading_price': max_trading_price})
            if args.rl_method == 'dqn':
                learner = DQNLearner(**{**common_params, 
                    'value_network_path': value_network_path})
            elif args.rl_method == 'pg':
                learner = PolicyGradientLearner(**{**common_params, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'ac':
                learner = ActorCriticLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'a2c':
                learner = A2CLearner(**{**common_params, 
                    'value_network_path': value_network_path, 
                    'policy_network_path': policy_network_path})
            elif args.rl_method == 'monkey':
                common_params['net'] = args.rl_method
                common_params['num_epoches'] = 10
                common_params['start_epsilon'] = 1
                learning = False
                learner = ReinforcementLearner(**common_params)
        else:
            list_stock_code.append(stock_code)
            list_chart_data.append(chart_data)
            list_training_data.append(training_data)
            list_min_trading_price.append(min_trading_price)
            list_max_trading_price.append(max_trading_price)

    if args.rl_method == 'a3c':
        learner = A3CLearner(**{
            **common_params, 
            'list_stock_code': list_stock_code, 
            'list_chart_data': list_chart_data, 
            'list_training_data': list_training_data,
            'list_min_trading_price': list_min_trading_price, 
            'list_max_trading_price': list_max_trading_price,
            'value_network_path': value_network_path, 
            'policy_network_path': policy_network_path})
    
    assert learner is not None

    if args.mode in ['train', 'test', 'update']:
        learner.run(learning=learning)
        if args.mode in ['train', 'update']:
            learner.save_models()
    elif args.mode == 'predict':
        learner.predict()
