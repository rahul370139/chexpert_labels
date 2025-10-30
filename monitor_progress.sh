#!/bin/bash
# Monitor progress of 1000-image evaluation

SSH_HOST="bilbouser@100.77.217.18"
REMOTE_DIR="~/chexagent_chexpert_eval"
LOG_FILE="evaluation_1000_improved.log"
OUTPUT_FILE="hybrid_ensemble_1000_improved.csv"

echo "📊 Monitoring Evaluation Progress"
echo "=================================="
echo ""

while true; do
    CURRENT=$(ssh $SSH_HOST "cd $REMOTE_DIR && grep -c 'Processing' $LOG_FILE 2>/dev/null | tail -1 || echo '0'")
    TOTAL=1000
    
    if [ "$CURRENT" -gt 0 ]; then
        PERCENT=$(echo "scale=1; $CURRENT * 100 / $TOTAL" | bc)
        
        # Check if process is still running
        RUNNING=$(ssh $SSH_HOST "ps aux | grep 'smart_ensemble.py.*1000_absolute' | grep -v grep | wc -l")
        
        # Get latest log entries
        LATEST=$(ssh $SSH_HOST "cd $REMOTE_DIR && tail -3 $LOG_FILE | grep -E 'Processing|Final positives' | tail -2")
        
        # Check file size if exists
        FILESIZE=$(ssh $SSH_HOST "cd $REMOTE_DIR && ls -lh $OUTPUT_FILE 2>/dev/null | awk '{print \$5}' || echo 'Not created'")
        
        clear
        echo "📊 Evaluation Progress Monitor"
        echo "=============================="
        echo ""
        echo "Progress: $CURRENT / $TOTAL ($PERCENT%)"
        
        if [ "$RUNNING" -gt 0 ]; then
            echo "Status: ✅ Running"
        else
            echo "Status: ⚠️  Process may have stopped"
        fi
        
        echo "Output file: $OUTPUT_FILE ($FILESIZE)"
        echo ""
        echo "Latest activity:"
        echo "$LATEST" | head -2
        echo ""
        
        if [ "$CURRENT" -eq "$TOTAL" ]; then
            echo "🎉 Evaluation complete!"
            break
        fi
        
        # Estimate time remaining
        if [ "$CURRENT" -gt 10 ]; then
            # Rough estimate: ~3-4 seconds per image
            REMAINING=$(( ($TOTAL - $CURRENT) * 4 ))
            MINUTES=$(( $REMAINING / 60 ))
            echo "Estimated time remaining: ~${MINUTES} minutes"
        fi
    else
        echo "Waiting for process to start..."
    fi
    
    sleep 30  # Update every 30 seconds
done

