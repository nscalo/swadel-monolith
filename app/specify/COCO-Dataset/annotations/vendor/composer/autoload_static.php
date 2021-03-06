<?php

// autoload_static.php @generated by Composer

namespace Composer\Autoload;

class ComposerStaticInitef6fbeb76b1c298f298dccb1f536fcf8
{
    public static $prefixLengthsPsr4 = array (
        'S' => 
        array (
            'Symfony\\Component\\OptionsResolver\\' => 34,
        ),
        'I' => 
        array (
            'Ivory\\Serializer\\' => 17,
        ),
        'D' => 
        array (
            'Doctrine\\Instantiator\\' => 22,
            'Doctrine\\Common\\Lexer\\' => 22,
        ),
    );

    public static $prefixDirsPsr4 = array (
        'Symfony\\Component\\OptionsResolver\\' => 
        array (
            0 => __DIR__ . '/..' . '/symfony/options-resolver',
        ),
        'Ivory\\Serializer\\' => 
        array (
            0 => __DIR__ . '/..' . '/egeloen/serializer/src',
        ),
        'Doctrine\\Instantiator\\' => 
        array (
            0 => __DIR__ . '/..' . '/doctrine/instantiator/src/Doctrine/Instantiator',
        ),
        'Doctrine\\Common\\Lexer\\' => 
        array (
            0 => __DIR__ . '/..' . '/doctrine/lexer/lib/Doctrine/Common/Lexer',
        ),
    );

    public static function getInitializer(ClassLoader $loader)
    {
        return \Closure::bind(function () use ($loader) {
            $loader->prefixLengthsPsr4 = ComposerStaticInitef6fbeb76b1c298f298dccb1f536fcf8::$prefixLengthsPsr4;
            $loader->prefixDirsPsr4 = ComposerStaticInitef6fbeb76b1c298f298dccb1f536fcf8::$prefixDirsPsr4;

        }, null, ClassLoader::class);
    }
}
