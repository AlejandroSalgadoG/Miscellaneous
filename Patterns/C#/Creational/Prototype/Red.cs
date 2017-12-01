public class Red : Color{

    public Red(string info){
        this.info = info;
    }

    public override string getInfo(){
        return info;
    }

    public override Color clone(){
        return new Red(info);
    }

}
