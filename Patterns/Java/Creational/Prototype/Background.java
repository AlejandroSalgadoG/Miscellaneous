public class Background{

    public void setBackground(Color color){
        Color colorCloned = color.clone();
        System.out.println( "The background is " + colorCloned.getInfo() );
    }


}
