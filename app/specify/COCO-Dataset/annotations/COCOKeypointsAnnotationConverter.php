<?php

require 'vendor/autoload.php';

use Ivory\Serializer\Format;
use Ivory\Serializer\Serializer;

class COCOKeypointsAnnotationConverter {

    private $valFile = null;
    private $trainFile = null;
    public $valXmlFile = null;
    public $trainXmlFile = null;
    public $valContents = array();
    public $trainContents = array();
    
    public function __construct($year, $path = "./annfiles") {
        if(!is_int($year)) {
            throw new \Exception("year variable is not an integer");
        }
        if(file_exists($path . "/person_keypoints_val" . $year . ".json")) {
            $this->valFile = $path . "/person_keypoints_val" . $year . ".json";
            $this->valXmlFile = $path . "/person_keypoints_val" . $year . ".xml";
        }
        if(file_exists($path . "/person_keypoints_train" . $year . ".json")) {
            $this->trainFile = $path . "/person_keypoints_train" . $year . ".json";
            $this->trainXmlFile = $path . "/person_keypoints_train" . $year . ".xml";
        }
    }

    public function readValFile()
    {
        $this->valContents = json_decode(file_get_contents($this->valFile));
    }

    public function readTrainFile()
    {
        $this->trainContents = json_decode(file_get_contents($this->trainFile));
    }

    public function groupPoints($annotations) {
        $grouped_points = [];
        
        array_map(function($annotation, $index) use (&$grouped_points) {
            $keypoints = $annotation->keypoints;
            $image_id = $annotation->image_id;
            $grouped_points[$image_id] = $keypoints;
        }, $annotations, range(0,count($annotations)-1));
        
        return $grouped_points;
    }

    public function arrayToObject($array) 
    {
        if (!is_array($array)) {
            return $array;
        }
        
        $object = new stdClass();
        if (is_array($array) && count($array) > 0) {
            foreach ($array as $name=>$value) {
                $name = strtolower(trim($name));
                if (!empty($name)) {
                    $object->$name = $this->arrayToObject($value);
                }
            }
            return $object;
        }
        else {
            return FALSE;
        }
    }

    public function extractKeypoints($grouped_points) {
        $node = new \stdClass();
        $annotation = new \stdClass();
        $meta = array(
            'task' => ['size' => filesize($this->valFile)]
        );
        $annotation->meta = $this->arrayToObject($meta);
        $images = [];

        $context = $this;
        
        array_map(function($grouped_point) use (&$images, $context) {
            $image = new \stdClass();
            $image->points = $context->arrayToObject($grouped_point);
            array_push($images, $image);
        }, $grouped_points);

        $annotation->images = $images;
        $node->annotations = $annotation;
        return $node;
    }

    public static function convertToPoseXml($xmlFile, \stdClass $contents)
    {
        if(empty($contents)) {
            throw new \Exception("Contents array is empty");
        }
        $annotations = $contents->annotations;

        $serializer = new Serializer();

        $xml = $serializer->serialize(
            $annotations, 
            Format::XML
        );

        file_put_contents($xmlFile, $xml);

        echo "The file has been written to " . $xmlFile;
        exit(0);
    }
}

$keypoints = new COCOKeypointsAnnotationConverter(2017);

$keypoints->readValFile();

$grouped_points = $keypoints->groupPoints($keypoints->valContents->annotations);
$node = $keypoints->extractKeypoints($grouped_points);

COCOKeypointsAnnotationConverter::convertToPoseXml($keypoints->valXmlFile, $node);

echo "Done executing";
