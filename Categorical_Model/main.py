import logistic
import RF
import Temp_Pred_mlp
import ensemble
import msvcrt
import matplotlib

if __name__ == "__main__":
    matplotlib.use('Agg')
    print("Logistic Regression (Baseline Model) Is Processing...")
    logistic.main()
    print('Press Any Key to Continue...')
    msvcrt.getch()
    print('-'*80)
    print("Random Forest Is Processing...")
    RF.main()
    print('Press Any Key to Continue...')
    msvcrt.getch()
    print('-'*80)
    print("Multilayer Perceptron Is Processing...")
    Temp_Pred_mlp.main()
    print('Press Any Key to Continue...')
    msvcrt.getch()
    print('-'*80)
    print("Ensemble Model Is Processing...")
    ensemble.main()
    print('Press Any Key to End')
    msvcrt.getch()
