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
import type { SchemaDefinition } from './schema-definition';

/**
 * 
 * @export
 * @interface NodeDataInput
 */
export interface NodeDataInput {
    /**
     * 
     * @type {SchemaDefinition}
     * @memberof NodeDataInput
     */
    'schema': SchemaDefinition;
    /**
     * Default values and template references
     * @type {{ [key: string]: any; }}
     * @memberof NodeDataInput
     */
    'values'?: { [key: string]: any; };
}

