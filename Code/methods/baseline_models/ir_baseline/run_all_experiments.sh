echo "DESCRIPTION"
# description

python evaluate_retrieval.py --use_move -src=che-eng.0 > outputs/move.test.che-eng.0.outputs
./multi-bleu.perl -lc data/test.che-eng.0.en < outputs/move.test.che-eng.0.outputs


python evaluate_retrieval.py --use_move --use_threat -src=che-eng.0 > outputs/move.threat.test.che-eng.0.outputs
./multi-bleu.perl -lc data/test.che-eng.0.en < outputs/move.threat.test.che-eng.0.outputs


python evaluate_retrieval.py --use_score --use_move --use_threat -src=che-eng.0 > outputs/move.threat.score.test.che-eng.0.outputs
./multi-bleu.perl -lc data/test.che-eng.0.en < outputs/move.threat.score.test.che-eng.0.outputs


#####################################################

echo "QUALITY"


# quality

python evaluate_retrieval.py --use_move -src=che-eng.1 > outputs/move.test.che-eng.1.outputs
./multi-bleu.perl -lc data/test.che-eng.1.en < outputs/move.test.che-eng.1.outputs

python evaluate_retrieval.py --use_move --use_threat -src=che-eng.1 > outputs/move.threat.test.che-eng.1.outputs
./multi-bleu.perl -lc data/test.che-eng.1.en < outputs/move.threat.test.che-eng.1.outputs

python evaluate_retrieval.py --use_score --use_move --use_threat -src=che-eng.1 > outputs/move.threat.score.test.che-eng.1.outputs
./multi-bleu.perl -lc data/test.che-eng.1.en < outputs/move.threat.score.test.che-eng.1.outputs


#####################################################

echo "PLANNING - COMPARITIVE"


# planning-comparitive

python evaluate_retrieval.py --use_move -src=che-eng.2.comparitive > outputs/move.test.che-eng.2.comparitive.outputs
./multi-bleu.perl -lc data/test.che-eng.2.comparitive.en < outputs/move.test.che-eng.2.comparitive.outputs


python evaluate_retrieval.py --use_move --use_threat -src=che-eng.2.comparitive > outputs/move.threat.test.che-eng.2.comparitive.outputs
./multi-bleu.perl -lc data/test.che-eng.2.comparitive.en < outputs/move.threat.test.che-eng.2.comparitive.outputs


python evaluate_retrieval.py --use_score --use_move --use_threat -src=che-eng.2.comparitive > outputs/move.threat.score.test.che-eng.2.comparitive.outputs
./multi-bleu.perl -lc data/test.che-eng.2.comparitive.en < outputs/move.threat.score.test.che-eng.2.comparitive.outputs


#####################################################
