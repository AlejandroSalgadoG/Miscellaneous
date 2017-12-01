package com.serviceapp;

import android.app.Activity;
import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

public class ServiceApp extends Activity {

	private Intent intent;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_service_app);
		
		intent = new Intent(this, TimeService.class);

		final Button start = (Button) findViewById(R.id.start);
		final Button stop = (Button) findViewById(R.id.stop);
		
		start.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				startService(intent); 
			}
		});

		stop.setOnClickListener(new View.OnClickListener() {
			@Override
			public void onClick(View v) {
				stopService(intent); 
			}
		});

	}

	private BroadcastReceiver receiver = new BroadcastReceiver() {
		public void onReceive(Context content, final Intent intent) {
			final TextView txtOut = (TextView) findViewById(R.id.text);
			String data = intent.getStringExtra("time").toString();
			txtOut.setText(data);
		}
	};

	@Override
	protected void onPause() {
		super.onPause();
		unregisterReceiver(receiver); 
	}

	@Override
	protected void onResume() {
		super.onResume();
		registerReceiver(receiver, new IntentFilter(TimeService.PATH));
	}

}
