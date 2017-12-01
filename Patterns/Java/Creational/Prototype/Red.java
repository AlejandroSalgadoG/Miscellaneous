public class Red extends Color{

    public Red(String info){
        this.info = info;
    }

    public String getInfo(){
        return info;
    }

    public Color clone(){
        return new Red(info);
    }

}
