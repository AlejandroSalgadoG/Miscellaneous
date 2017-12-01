package com.ba_lifecyrcle;

import java.text.SimpleDateFormat;
import java.util.Calendar;
import android.os.Bundle;
import android.app.Activity;
import android.content.SharedPreferences;
import android.widget.TextView;

public class MainActivity extends Activity {
	public static final String date = "HH:mm:ss";
	String status = "";

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		
		SharedPreferences oSettings = getSharedPreferences("app_settings", 0);
		/*
		 * Devulve una instancia de SharedPreferences con el archivo 
		 * que se llama como el primer parametro, si no existe se crea,
		 * el segundo parametro indica si es privado o publico, 
		 * el 0 es lo mismo q privado 
		 */
		status = oSettings.getString("screen_msg", ""); 
		/*
		 * lee la variable, primer parametro, en caso de no existir
		 * retorna el valor del segundo parametro, "".
		*/
		setStatus("onCreate");
	}
	
	private String getCurrentDateTime() {
		Calendar cal = Calendar.getInstance();
		SimpleDateFormat sdf = new SimpleDateFormat(date);
		return sdf.format(cal.getTime());
	}

	void setStatus(String sEvent) {
		if (sEvent.equalsIgnoreCase("onCreate")) 
			status += "===============================\n";

		status += "In " + sEvent + "() at: " + getCurrentDateTime() + "\n";
		final TextView tOut = (TextView) findViewById(R.id.text);
		tOut.setText(status);
	}

	@Override
	protected void onDestroy() {
		super.onDestroy();
		setStatus("onDestroy");
	}

	@Override
	protected void onPause() {
		super.onPause();
		setStatus("onPause");
	}

	@Override
	protected void onRestart() {
		super.onRestart();
		setStatus("onRestart");
	}

	@Override
	protected void onResume() {
		super.onResume();
		setStatus("onResume");
	}

	@Override
	protected void onStart() {
		super.onStart();
		setStatus("onStart");
	}

	@Override
	protected void onStop() {
		super.onStop();
		setStatus("onStop");

		SharedPreferences oSettings = getSharedPreferences("app_settings", 0);
		SharedPreferences.Editor oEdit = oSettings.edit();
		/*
		 * Obtiene una instancia de editor para editar el archivo del 
		 * SharedPreferences
		 */
		oEdit.putString("screen_msg", status);
		/*
		 * guarda el valor del sundo parametro en el primero, 
		 * y el primero es almacenado en el archivo
		 */
		oEdit.commit();
		/*
		 * vuelve permanente lo echo con el metodo putString()
		 */
	}

}
