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
import type { ModelSpec } from './model-spec';

/**
 * 
 * @export
 * @interface ModelConfig
 */
export interface ModelConfig {
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'name'?: string;
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'completion_dialect'?: string;
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'embedding_dialect'?: string;
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'rerank_dialect'?: string;
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'label'?: string;
    /**
     * 
     * @type {boolean}
     * @memberof ModelConfig
     */
    'allow_custom_base_url'?: boolean;
    /**
     * 
     * @type {string}
     * @memberof ModelConfig
     */
    'base_url'?: string;
    /**
     * 
     * @type {Array<ModelSpec>}
     * @memberof ModelConfig
     */
    'embedding'?: Array<ModelSpec>;
    /**
     * 
     * @type {Array<ModelSpec>}
     * @memberof ModelConfig
     */
    'completion'?: Array<ModelSpec>;
    /**
     * 
     * @type {Array<ModelSpec>}
     * @memberof ModelConfig
     */
    'rerank'?: Array<ModelSpec>;
}

