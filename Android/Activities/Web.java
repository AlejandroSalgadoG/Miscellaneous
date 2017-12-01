package com.activities;

import android.app.Activity;
import android.os.Bundle;
import android.webkit.WebView;
import android.webkit.WebViewClient;

public class Web extends Activity {

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_web);
		
		WebView web = (WebView) findViewById(R.id.webView);
		web.getSettings().setJavaScriptEnabled(true);
		web.loadUrl("https://www.bing.com");
		web.setWebViewClient(new Client());
	}
	
	private class Client extends WebViewClient{
		@Override
		public boolean shouldOverrideUrlLoading(WebView view, String url){
			view.loadUrl(url);
			return true;
		}
	}
	
}
