/* tslint:disable */
/* eslint-disable */
/**
 * ApeRAG API
 * ApeRAG API Documentation
 *
 * The version of the OpenAPI document: 1.0.0
 * 
 *
 * NOTE: This class is auto generated by OpenAPI Generator (https://openapi-generator.tech).
 * https://openapi-generator.tech
 * Do not edit the class manually.
 */


// May contain unused imports in some cases
// @ts-ignore
import type { ConfigAuth } from './config-auth';

/**
 * 
 * @export
 * @interface Config
 */
export interface Config {
    /**
     * Whether the admin user exists
     * @type {boolean}
     * @memberof Config
     */
    'admin_user_exists'?: boolean;
    /**
     * 
     * @type {ConfigAuth}
     * @memberof Config
     */
    'auth'?: ConfigAuth;
}

