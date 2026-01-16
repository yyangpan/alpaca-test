# Running OTC Trading Bot on AWS EC2

This guide explains how to set up and run the OTC trading bot on AWS EC2.

## Prerequisites

- AWS EC2 instance (Amazon Linux 2, Ubuntu, or similar)
- Python 3.8 or higher
- SSH access to your EC2 instance

## Quick Start

### 1. Upload Files to EC2

Upload the following files to your EC2 instance:
- `otc_trading_bot.py` - Main trading bot script
- `requirements.txt` - Python dependencies
- `setup.sh` - Setup script
- `run_daily.sh` - Daily runner script

You can use `scp` to upload files:
```bash
scp -i your-key.pem otc_trading_bot.py ec2-user@your-ec2-ip:~/
scp -i your-key.pem requirements.txt ec2-user@your-ec2-ip:~/
scp -i your-key.pem setup.sh ec2-user@your-ec2-ip:~/
scp -i your-key.pem run_daily.sh ec2-user@your-ec2-ip:~/
```

### 2. SSH into EC2

```bash
ssh -i your-key.pem ec2-user@your-ec2-ip
```

### 3. Run Setup

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Check Python version
- Create a virtual environment (optional but recommended)
- Install required Python packages
- Make scripts executable

### 4. Test the Bot

First, test the bot manually:

```bash
# Activate virtual environment
source venv/bin/activate

# View help
python3 otc_trading_bot.py --help

# View account information
python3 otc_trading_bot.py --account

# View current positions
python3 otc_trading_bot.py --positions

# Run with auto-detected OTC symbols
python3 otc_trading_bot.py

# Run with specific symbols
python3 otc_trading_bot.py --symbols SYMBOL1 SYMBOL2 SYMBOL3

# Run with symbols from file
python3 otc_trading_bot.py --symbols-file otc_symbols.txt
```

## Setting Up Automated Daily Execution

### Option 1: Using Cron (Recommended)

Cron is the simplest way to schedule daily execution.

1. **Make the daily runner executable:**
```bash
chmod +x run_daily.sh
```

2. **Edit crontab:**
```bash
crontab -e
```

3. **Add a cron job to run daily at 9:30 AM ET (market open):**

For Amazon Linux 2 / RHEL:
```cron
30 9 * * 1-5 cd /home/ec2-user/alpace-test && /home/ec2-user/alpace-test/run_daily.sh >> /home/ec2-user/alpace-test/logs/cron.log 2>&1
```

For Ubuntu:
```cron
30 9 * * 1-5 cd /home/ubuntu/alpace-test && /home/ubuntu/alpace-test/run_daily.sh >> /home/ubuntu/alpace-test/logs/cron.log 2>&1
```

**Note:** Adjust the path and username based on your EC2 setup. The time is in server time (UTC). ET is UTC-5 (or UTC-4 during DST).

To run at 9:30 AM ET (14:30 UTC):
```cron
30 14 * * 1-5 cd /home/ec2-user/alpace-test && /home/ec2-user/alpace-test/run_daily.sh >> /home/ec2-user/alpace-test/logs/cron.log 2>&1
```

4. **Verify cron job:**
```bash
crontab -l
```

### Option 2: Using Systemd Timer

For more advanced scheduling, you can use systemd timers.

1. **Create a systemd service file** (`/etc/systemd/system/otc-trading.service`):
```ini
[Unit]
Description=OTC Trading Bot
After=network.target

[Service]
Type=oneshot
User=ec2-user
WorkingDirectory=/home/ec2-user/alpace-test
ExecStart=/home/ec2-user/alpace-test/run_daily.sh
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

2. **Create a systemd timer file** (`/etc/systemd/system/otc-trading.timer`):
```ini
[Unit]
Description=Run OTC Trading Bot Daily
Requires=otc-trading.service

[Timer]
OnCalendar=Mon-Fri 14:30:00
Timezone=America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

3. **Enable and start the timer:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable otc-trading.timer
sudo systemctl start otc-trading.timer
sudo systemctl status otc-trading.timer
```

## Providing OTC Symbols

Since Alpaca's OTC coverage may be limited, you can provide your own list of OTC symbols.

### Method 1: Symbols File

1. Create `otc_symbols.txt`:
```bash
cp otc_symbols.txt.example otc_symbols.txt
nano otc_symbols.txt
```

2. Add one symbol per line:
```
SYMBOL1
SYMBOL2
SYMBOL3
```

3. Run with symbols file:
```bash
python3 otc_trading_bot.py --symbols-file otc_symbols.txt
```

4. Update `run_daily.sh` to use the symbols file:
```bash
# Edit run_daily.sh and uncomment/add:
python3 otc_trading_bot.py --symbols-file otc_symbols.txt
```

### Method 2: Command Line

```bash
python3 otc_trading_bot.py --symbols SYMBOL1 SYMBOL2 SYMBOL3
```

## Monitoring

### View Logs

The bot creates a log file `otc_trading.log`:
```bash
tail -f otc_trading.log
```

### Check Buy Schedules

Buy schedules are saved in `buy_schedules.json`:
```bash
cat buy_schedules.json
```

### View Account Status

```bash
python3 otc_trading_bot.py --account
```

### View Positions

```bash
python3 otc_trading_bot.py --positions
```

### Check Profit Targets Only

```bash
python3 otc_trading_bot.py --check-only
```

## Troubleshooting

### Bot Not Running

1. **Check if cron is running:**
```bash
sudo service crond status  # Amazon Linux 2
sudo service cron status   # Ubuntu
```

2. **Check cron logs:**
```bash
tail -f logs/cron.log
```

3. **Test manually:**
```bash
source venv/bin/activate
python3 otc_trading_bot.py
```

### Python Not Found

If you get "python3: command not found", ensure you're using the correct path:
```bash
which python3
```

Update `run_daily.sh` to use the full path if needed.

### Virtual Environment Issues

If using virtual environment, ensure it's activated in `run_daily.sh`:
```bash
# In run_daily.sh, ensure this line exists:
source venv/bin/activate
```

### Permission Denied

Make scripts executable:
```bash
chmod +x run_daily.sh
chmod +x otc_trading_bot.py
```

### Alpaca API Errors

- Check API credentials in `otc_trading_bot.py`
- Verify internet connection
- Check Alpaca service status
- Review `otc_trading.log` for detailed error messages

## Best Practices

1. **Test First**: Always test the bot manually before setting up automation
2. **Monitor Logs**: Regularly check `otc_trading.log` for errors
3. **Backup Data**: Regularly backup `buy_schedules.json`
4. **Use Paper Trading**: Start with paper trading credentials (already configured)
5. **Set Alerts**: Consider setting up email/SNS alerts for errors
6. **Market Hours**: Ensure cron runs during market hours
7. **Time Zone**: Be aware of time zone differences between EC2 (UTC) and market hours (ET)

## Security Notes

⚠️ **Important**: The script contains API credentials. Ensure proper file permissions:
```bash
chmod 600 otc_trading_bot.py
```

For production, consider using environment variables or AWS Secrets Manager for API credentials.

## Updating the Bot

To update the bot:
1. Upload new version of `otc_trading_bot.py`
2. Test manually
3. Restart cron/systemd timer if needed

## Support

For issues:
1. Check `otc_trading.log` for errors
2. Verify API credentials
3. Test API connection: `python3 otc_trading_bot.py --account`
4. Review Alpaca API documentation: https://alpaca.markets/docs/

