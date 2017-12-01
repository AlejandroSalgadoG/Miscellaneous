package com.serviceapp;

import java.util.Calendar;

import android.app.Service;
import android.content.Intent;
import android.os.Handler;
import android.os.IBinder;

public class TimeService extends Service {

	public static final String PATH = "com.example.ba_services.TimerService";
	private final Handler handler = new Handler();
	private Intent intent;
	private long start = 0;

	@Override
	public void onCreate() {
		super.onCreate();
		
		intent = new Intent(PATH);
		
		Calendar time = Calendar.getInstance();
		start = time.getTimeInMillis();
		
		handler.removeCallbacks(sendToUI);
		handler.postDelayed(sendToUI, 1000);
	}

	private Runnable sendToUI = new Runnable() {
		public void run() {
			getCurrentData();
			handler.postDelayed(this, 1000);
		}
	};

	private void getCurrentData() {
		Calendar time = Calendar.getInstance();
		long mill = time.getTimeInMillis();
		long diff = mill - start;
		long ans = diff / 1000; 

		intent.putExtra("time", Long.toString(ans));
		sendBroadcast(intent);
	}

	@Override
	public void onDestroy() {
		handler.removeCallbacks(sendToUI);
		super.onDestroy();
	}

	@Override
	public IBinder onBind(Intent arg0) {
		return null;
	}
}
